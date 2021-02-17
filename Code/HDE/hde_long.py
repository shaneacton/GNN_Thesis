import time
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import ReLU, CrossEntropyLoss
from torch_geometric.nn import GATConv
from transformers import LongformerTokenizerFast, BatchEncoding

from Code.Config import sysconf, vizconf, gcc
from Code.Data.Text.longformer_embedder import LongformerEmbedder
from Code.HDE.Graph.graph import HDEGraph
from Code.HDE.Graph.graph_utils import add_doc_nodes, add_entity_nodes, add_candidate_nodes, \
    connect_candidates_and_entities, connect_unconnected_entities, connect_entity_mentions, similar
from Code.HDE.coattention import Coattention
from Code.HDE.gnn_stack import GNNStack
from Code.HDE.scorer import HDEScorer
from Code.HDE.summariser import Summariser, NODE_TYPE_MAP
from Code.HDE.visualiser import render_graph
from Code.Training import device
from Code.Training.Utils.initialiser import get_tokenizer
from Code.constants import CANDIDATE, DOCUMENT, QUERY

SOURCE_TYPE_MAP = {QUERY: 0, DOCUMENT: 1, CANDIDATE: 2}


class HDELongStack(nn.Module):

    def __init__(self, num_layers=2, hidden_size=-1, heads=1, dropout=0.1, name=None, use_contextual_embs=False):
        """hidden size = -1 leaves hidden to be same as pretrained emb dim"""
        super().__init__()
        self.use_contextual_embs = use_contextual_embs
        self.name = name
        self.tokeniser: LongformerTokenizerFast = get_tokenizer()
        self.token_embedder = LongformerEmbedder(out_features=hidden_size)
        hidden_size = self.token_embedder.out_features

        if not use_contextual_embs:  # fine tune embs
            for param in self.non_ctx_embedder.parameters():
                param.requires_grad = True

            self.coattention = Coattention(hidden_size)

        self.summariser = Summariser(hidden_size)
        self.relu = ReLU()

        self.gnn = GNNStack(GATConv, num_layers, hidden_size, hidden_size, dropout=dropout, heads=heads)

        self.candidate_scorer = HDEScorer(hidden_size)
        self.entity_scorer = HDEScorer(hidden_size)

        self.loss_fn = CrossEntropyLoss()
        self.last_example = -1
        self.last_epoch = -1

    @property
    def non_ctx_embedder(self):
        return self.token_embedder.longformer.embeddings

    def pass_non_ctx_embedder(self, encoding, type):
        ids = torch.tensor(encoding["input_ids"]).long().to(device).view(1, -1)
        type_ids = self.summariser.get_type_tensor(type, ids.size(-1), type_map=SOURCE_TYPE_MAP)
        embs = self.non_ctx_embedder(input_ids=ids, token_type_ids=type_ids)
        return embs

    def forward(self, supports: List[str], query: str, candidates: List[str], answer=None):
        """
            nodes are created for each support, as well as each candidate and each context entity
            nodes are concattenated as follows: supports, entities, candidates

            nodes are connected according to the HDE paper
            the graph is converted to a pytorch geometric datapoint
        """
        support_encodings, candidate_encodings = self.get_encodings(supports, query, candidates)
        t = time.time()

        if self.use_contextual_embs:
            support_embeddings = [self.token_embedder(sup_enc) for sup_enc in support_encodings]
        else:
            """
                must do coattention between context and query to get local contexts 
                and to incorporate query info into nodes
            """
            support_embeddings = [self.pass_non_ctx_embedder(sup_enc, DOCUMENT) for sup_enc in support_encodings]

            query_encoding = self.tokeniser(query)
            query_embedding = self.pass_non_ctx_embedder(query_encoding, QUERY)
            support_embeddings = [self.coattention(emb, query_embedding) for emb in support_embeddings]
        if sysconf.print_times:
            print("got support embeddings in", (time.time() - t))

        t = time.time()
        cand_embeddings = [self.pass_non_ctx_embedder(cand_enc, CANDIDATE) for cand_enc in candidate_encodings]
        candidate_summaries = [self.summariser(emb, CANDIDATE) for emb in cand_embeddings]

        if sysconf.print_times:
            print("got candidate embeddings in", (time.time() - t))
        support_summaries = [self.summariser(sup_emb, DOCUMENT) for sup_emb in support_embeddings]

        t = time.time()

        ent_token_spans, ent_summaries, = get_entities(self.summariser, support_embeddings, support_encodings, supports)

        if sysconf.print_times:
            print("got ents in", (time.time() - t))

        graph = self.create_graph(candidates, ent_token_spans, supports, support_encodings)
        if vizconf.visualise_graphs:
            render_graph(graph)
            if vizconf.exit_after_first_viz:
                exit()

        t = time.time()

        x = torch.cat(support_summaries + ent_summaries + candidate_summaries)

        edge_index = graph.edge_index
        # print("edge index:", edge_index.size(), edge_index.type())

        x = self.gnn(x, edge_index)

        if sysconf.print_times:
            print("passed gnn in", (time.time() - t))
        t = time.time()

        # x has now been transformed by the GNN layers. Must map to  a prob dist over candidates

        cand_idxs = graph.candidate_nodes

        cand_embs = x[cand_idxs[0]: cand_idxs[-1] + 1, :]
        cand_probs = self.candidate_scorer(cand_embs).view(len(graph.candidate_nodes))

        ent_probs = []
        for c, cand in enumerate(candidates):
            """find all entities which match this candidate, and score them each, returning the maximum score"""
            cand_node = graph.ordered_nodes[cand_idxs[c]]
            linked_ent_nodes = set()
            for ent_text in graph.entity_text_to_nodes.keys():  # find all entities with similar text
                if similar(cand_node.text, ent_text):
                    linked_ent_nodes.update(graph.entity_text_to_nodes[ent_text])

            if len(linked_ent_nodes) == 0:
                max_prob = torch.tensor(0.).to(device)
            else:
                linked_ent_nodes = sorted(list(linked_ent_nodes))  # node ids of all entities linked to this candidate
                linked_ent_nodes = torch.tensor(linked_ent_nodes).to(device).long()
                ent_embs = torch.index_select(x, dim=0, index=linked_ent_nodes)

                cand_ent_probs = self.entity_scorer(ent_embs)
                max_prob = torch.max(cand_ent_probs)
            ent_probs += [max_prob]
        ent_probs = torch.stack(ent_probs, dim=0)

        final_probs = cand_probs + ent_probs
        pred_id = torch.argmax(final_probs)
        pred_ans = candidates[pred_id]

        if sysconf.print_times:
            print("passed output model in", (time.time() - t))

        if answer is not None:
            ans_id = candidates.index(answer)
            probs = final_probs.view(1, -1)  # batch dim
            ans = torch.tensor([ans_id]).to(device)
            loss = self.loss_fn(probs, ans)

            return loss, pred_ans

        return pred_ans

    def get_encodings(self, supports: List[str], query: str, candidates: List[str]) \
            -> Tuple[List[BatchEncoding], List[BatchEncoding]]:
        supports = [s[:gcc.max_context_chars] for s in supports]
        if self.use_contextual_embs:
            support_encodings = [self.tokeniser(support, query) for support in supports]
        else:  # non contextual embs need a coattention module between query and context
            support_encodings = [self.tokeniser(support) for support in supports]

        candidate_encodings = [self.tokeniser(candidate) for candidate in candidates]
        return support_encodings, candidate_encodings

    def create_graph(self, candidates, ent_token_spans, supports, support_encodings):
        start_t = time.time()
        graph = HDEGraph()
        add_doc_nodes(graph, supports)
        add_entity_nodes(graph, supports, ent_token_spans, tokeniser=self.tokeniser, support_encodings=support_encodings)
        add_candidate_nodes(graph, candidates, supports)
        connect_candidates_and_entities(graph)
        connect_entity_mentions(graph)
        connect_unconnected_entities(graph)

        if sysconf.print_times:
            print("made full graph in", (time.time() - start_t))

        return graph
