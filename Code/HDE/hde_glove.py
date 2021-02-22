import time
from typing import List

import torch
from torch import nn
from torch.nn import ReLU, CrossEntropyLoss
from torch_geometric.nn import GATConv

from Code.Config.config import config
from Code.Embedding.Glove.glove_embedder import GloveEmbedder
from Code.GNNs.asymmetrical_gat import AsymGat
from Code.GNNs.r_gat import RGat
from Code.HDE.Transformers.coattention import Coattention
from Code.GNNs.gnn_stack import GNNStack
from Code.HDE.Graph.graph import HDEGraph
from Code.HDE.Graph.graph_utils import add_doc_nodes, add_entity_nodes, add_candidate_nodes, \
    connect_candidates_and_entities, connect_unconnected_entities, connect_entity_mentions, similar, \
    get_entity_summaries
from Code.HDE.scorer import HDEScorer
from Code.HDE.Transformers.summariser import Summariser
from Viz.graph_visualiser import render_graph
from Code.HDE.wikipoint import Wikipoint
from Code.Training import device
from Code.constants import CANDIDATE, DOCUMENT


class HDEGloveStack(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        self.name = config.model_name
        self.embedder = GloveEmbedder()
        self.hidden_size = config.hidden_size

        self.coattention = Coattention(**kwargs)
        self.summariser = Summariser(**kwargs)
        self.relu = ReLU()

        # self.gnn = GNNStack(RGat, num_types=7)
        self.gnn = GNNStack(GATConv)

        self.candidate_scorer = HDEScorer(config.hidden_size)
        self.entity_scorer = HDEScorer(config.hidden_size)

        self.loss_fn = CrossEntropyLoss()
        self.last_example = -1
        self.last_epoch = -1

    def forward(self, example: Wikipoint, graph=None):
        """
            nodes are created for each support, as well as each candidate and each context entity
            nodes are concattenated as follows: supports, entities, candidates

            nodes are connected according to the HDE paper
            the graph is converted to a pytorch geometric datapoint
        """
        if graph is None:
            graph = self.create_graph(example)
        x = self.get_graph_features(example)

        edge_index = graph.edge_index
        num_edges = len(graph.unique_edges)
        if num_edges > config.max_edges != -1:
            raise TooManyEdges()

        t = time.time()
        x = self.gnn(x, edge_index)

        if config.print_times:
            print("passed gnn in", (time.time() - t))
        t = time.time()

        # x has now been transformed by the GNN layers. Must map to  a prob dist over candidates
        final_probs = self.pass_output_model(x, example, graph)
        pred_id = torch.argmax(final_probs)
        pred_ans = example.candidates[pred_id]

        if config.print_times:
            print("passed output model in", (time.time() - t))

        if example.answer is not None:
            ans_id = example.candidates.index(example.answer)
            probs = final_probs.view(1, -1)  # batch dim
            ans = torch.tensor([ans_id]).to(device)
            loss = self.loss_fn(probs, ans)

            return loss, pred_ans

        return pred_ans

    def get_graph_features(self, example):
        support_embeddings = self.get_query_aware_context_embeddings(example.supports, example.query)
        cand_embs = [self.embedder(cand) for cand in example.candidates]
        candidate_summaries = self.summariser(cand_embs, CANDIDATE)
        support_summaries = self.summariser(support_embeddings, DOCUMENT)

        t = time.time()

        ent_summaries = get_entity_summaries(example.ent_token_spans, support_embeddings, self.summariser)
        if config.print_times:
            print("got ents in", (time.time() - t))

        x = torch.cat(support_summaries + ent_summaries + candidate_summaries)

        return x

    def pass_output_model(self, x, example, graph):
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
        return final_probs

    def get_query_aware_context_embeddings(self, supports: List[str], query: str):
        support_embeddings = [self.embedder(sup) for sup in supports]
        # print("supps:", [s.size() for s in support_embeddings])
        pad_volume = max([s.size(1) for s in support_embeddings]) * len(support_embeddings)
        if pad_volume > config.max_pad_volume:
            raise PadVolumeOverflow()
        # print("pad vol:", pad_volume)
        query_emb = self.embedder(query)
        support_embeddings = self.coattention.batched_coattention(support_embeddings, query_emb)
        # support_embeddings = [self.coattention(se, query_emb) for se in support_embeddings]
        return support_embeddings

    def create_graph(self, example):
        start_t = time.time()
        graph = HDEGraph()
        add_doc_nodes(graph, example.supports)
        add_entity_nodes(graph, example.supports, example.ent_token_spans, glove_embedder=self.embedder)
        add_candidate_nodes(graph, example.candidates, example.supports)
        connect_candidates_and_entities(graph)
        connect_entity_mentions(graph)
        connect_unconnected_entities(graph)

        if config.print_times:
            print("made full graph in", (time.time() - start_t))

        if config.visualise_graphs:
            render_graph(graph)
            if config.exit_after_first_viz:
                exit()

        return graph


class PadVolumeOverflow(Exception):
    pass


class TooManyEdges(Exception):
    pass
