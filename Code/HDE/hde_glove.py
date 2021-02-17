import time
from typing import List

import torch
from torch import nn
from torch.nn import ReLU, CrossEntropyLoss
from torch_geometric.nn import GATConv

from Code.Config import sysconf, vizconf
from Code.HDE.Glove.glove_embedder import GloveEmbedder
from Code.HDE.coattention import Coattention
from Code.HDE.gnn_stack import GNNStack
from Code.HDE.Graph.graph import HDEGraph
from Code.HDE.Graph.graph_utils import add_doc_nodes, add_entity_nodes, add_candidate_nodes, \
    connect_candidates_and_entities, connect_unconnected_entities, connect_entity_mentions, similar, \
    get_entity_summaries
from Code.HDE.scorer import HDEScorer
from Code.HDE.summariser import Summariser
from Code.HDE.visualiser import render_graph
from Code.HDE.wikipoint import Wikipoint
from Code.Training import device
from Code.constants import CANDIDATE, DOCUMENT


class HDEGloveStack(nn.Module):

    def __init__(self, num_layers=2, hidden_size=100, embedded_dims=50, heads=1, dropout=0.1, name=None):
        super().__init__()
        self.name = name
        self.embedder = GloveEmbedder(dims=embedded_dims)

        self.coattention = Coattention(self.embedder.dims)
        self.summariser = Summariser(self.embedder.dims)
        self.relu = ReLU()

        self.gnn = GNNStack(GATConv, num_layers, self.embedder.dims, hidden_size, dropout=dropout, heads=heads)

        self.candidate_scorer = HDEScorer(hidden_size)
        self.entity_scorer = HDEScorer(hidden_size)

        self.loss_fn = CrossEntropyLoss()
        self.last_example = -1
        self.last_epoch = -1

    def forward(self, example: Wikipoint):
        """
            nodes are created for each support, as well as each candidate and each context entity
            nodes are concattenated as follows: supports, entities, candidates

            nodes are connected according to the HDE paper
            the graph is converted to a pytorch geometric datapoint
        """
        support_embeddings = self.get_query_aware_context_embeddings(example.supports, example.query)
        candidate_summaries = [self.summariser(self.embedder(cand), CANDIDATE) for cand in example.candidates]
        support_summaries = [self.summariser(sup_emb, DOCUMENT) for sup_emb in support_embeddings]

        t = time.time()

        # ent_token_spans, ent_summaries, = get_glove_entities(self.summariser, support_embeddings, example.supports, self.embedder)

        if sysconf.print_times:
            print("got ents in", (time.time() - t))

        graph = self.create_graph(example.candidates, example.ent_token_spans, example.supports)
        if vizconf.visualise_graphs:
            render_graph(graph)
            if vizconf.exit_after_first_viz:
                exit()

        ent_summaries = get_entity_summaries(example.ent_token_spans, support_embeddings, self.summariser)

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
        for c, cand in enumerate(example.candidates):
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
        pred_ans = example.candidates[pred_id]

        if sysconf.print_times:
            print("passed output model in", (time.time() - t))

        if example.answer is not None:
            ans_id = example.candidates.index(example.answer)
            probs = final_probs.view(1, -1)  # batch dim
            ans = torch.tensor([ans_id]).to(device)
            loss = self.loss_fn(probs, ans)

            return loss, pred_ans

        return pred_ans

    def get_query_aware_context_embeddings(self, supports: List[str], query: str):
        support_embeddings = [self.embedder(sup) for sup in supports]
        query_emb = self.embedder(query)
        support_embeddings = [self.coattention(se, query_emb) for se in support_embeddings]
        return support_embeddings

    def create_graph(self, candidates, ent_token_spans, supports):
        start_t = time.time()
        graph = HDEGraph()
        add_doc_nodes(graph, supports)
        add_entity_nodes(graph, supports, ent_token_spans, glove_embedder=self.embedder)
        add_candidate_nodes(graph, candidates, supports)
        connect_candidates_and_entities(graph)
        connect_entity_mentions(graph)
        connect_unconnected_entities(graph)

        if sysconf.print_times:
            print("made full graph in", (time.time() - start_t))

        return graph
