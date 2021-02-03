import time
from typing import List

import torch
from torch import nn
from torch.nn import ReLU, CrossEntropyLoss
from torch_geometric.nn import GATConv

from Code.Config import sysconf, vizconf
from Code.HDE.Glove.glove_embedder import GloveEmbedder
from Code.HDE.Glove.glove_utils import get_glove_entities
from Code.HDE.coattention import Coattention
from Code.HDE.gnn_stack import GNNStack
from Code.HDE.Graph.graph import HDEGraph
from Code.HDE.Graph.graph_utils import add_doc_nodes, add_entity_nodes, add_candidate_nodes, \
    connect_candidates_and_entities, connect_unconnected_entities, connect_entity_mentions, similar
from Code.HDE.scorer import HDEScorer
from Code.HDE.summariser import Summariser
from Code.HDE.visualiser import render_graph
from Code.Training import device


class HDEGloveStack(nn.Module):

    def __init__(self, num_layers=2, hidden_size=100, embedded_dims=50):
        super().__init__()
        self.embedder = GloveEmbedder(dims=embedded_dims)

        self.coattention = Coattention(self.embedder.dims)
        self.summariser = Summariser(self.embedder.dims)
        self.relu = ReLU()

        self.gnn = GNNStack(GATConv, num_layers, self.embedder.dims, hidden_size, heads=1)

        self.candidate_scorer = HDEScorer(hidden_size)
        self.entity_scorer = HDEScorer(hidden_size)

        self.loss_fn = CrossEntropyLoss()
        self.last_example = -1
        self.last_epoch = -1

    def forward(self, supports: List[str], query: str, candidates: List[str], answer=None):
        """
            nodes are created for each support, as well as each candidate and each context entity
            nodes are concattenated as follows: supports, entities, candidates

            nodes are connected according to the HDE paper
            the graph is converted to a pytorch geometric datapoint
        """
        t = time.time()

        support_embeddings = self.get_query_aware_context_embeddings(supports, query)

        if sysconf.print_times:
            print("got embeddings in", (time.time() - t))
        t = time.time()

        candidate_summaries = [self.summariser(self.embedder(cand)) for cand in candidates]
        support_summaries = [self.summariser(sup_emb) for sup_emb in support_embeddings]

        if sysconf.print_times:
            print("got summaries in", (time.time() - t))
        t = time.time()

        ent_token_spans, ent_summaries, = get_glove_entities(self.summariser, support_embeddings, supports, self.embedder)

        if sysconf.print_times:
            print("got ents in", (time.time() - t))
        t = time.time()

        graph = self.create_graph(candidates, ent_token_spans, supports)
        if vizconf.visualise_graphs:
            render_graph(graph)
            if vizconf.exit_after_first_viz:
                exit()

        if sysconf.print_times:
            print("made graph in", (time.time() - t))
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
        # print("cand probs:", cand_probs.size())

        ent_probs = []
        for c, cand in enumerate(candidates):
            """find all entities which match this candidate, and score them each, returning the maximum score"""
            cand_node = graph.ordered_nodes[cand_idxs[c]]
            linked_ent_nodes = set()
            for ent_text in graph.entity_text_to_nodes.keys():  # find all entities with similar text
                if similar(cand_node.text, ent_text):
                    # print("\tcand:", cand, "ents:", ent_text)
                    linked_ent_nodes.update(graph.entity_text_to_nodes[ent_text])

            if len(linked_ent_nodes) == 0:
                max_prob = torch.tensor(0.).to(device)
            else:
                linked_ent_nodes = sorted(list(linked_ent_nodes))  # node ids of all entities linked to this candidate
                linked_ent_nodes = torch.tensor(linked_ent_nodes).to(device).long()
                ent_embs = torch.index_select(x, dim=0, index=linked_ent_nodes)

                # print("ent embs:", ent_embs.size())
                cand_ent_probs = self.entity_scorer(ent_embs)
                # print("ent probs:", cand_ent_probs.size())
                max_prob = torch.max(cand_ent_probs)
            ent_probs += [max_prob]
            # print("max:", max_prob)
        ent_probs = torch.stack(ent_probs, dim=0)
        # print("ent probs:", ent_probs.size())

        final_probs = cand_probs + ent_probs
        pred_id = torch.argmax(final_probs)
        pred_ans = candidates[pred_id]

        if answer is not None:
            ans_id = candidates.index(answer)
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
        graph = HDEGraph()
        add_doc_nodes(graph, supports)
        add_entity_nodes(graph, supports, ent_token_spans, glove_embedder=self.embedder)
        add_candidate_nodes(graph, candidates, supports)
        connect_candidates_and_entities(graph)
        connect_entity_mentions(graph)
        connect_unconnected_entities(graph)
        return graph
