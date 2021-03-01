import time
from typing import List

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, ReLU
from torch_geometric.nn import GATConv

from Code.Embedding.string_embedder import StringEmbedder
from Code.GNNs.gnn_stack import GNNStack
from Code.HDE.Graph.graph_utils import similar, get_entity_summaries
from Code.HDE.Transformers.coattention import Coattention
from Code.HDE.Transformers.summariser import Summariser
from Code.HDE.scorer import HDEScorer
from Code.HDE.wikipoint import Wikipoint
from Code.Training import device
from Code.constants import DOCUMENT, CANDIDATE
from Config.config import conf


class HDEModel(nn.Module):

    def __init__(self, GNNClass=GATConv, **kwargs):
        super().__init__()
        self.name = conf.model_name
        self.hidden_size = conf.hidden_size

        self.coattention = Coattention(**kwargs)
        self.summariser = Summariser(**kwargs)

        self.relu = ReLU()

        # self.gnn = GNNStack(RGat, num_types=7)
        if GNNClass is not None:
            self.gnn = GNNStack(GNNClass)

        self.candidate_scorer = HDEScorer(conf.hidden_size)
        self.entity_scorer = HDEScorer(conf.hidden_size)

        self.loss_fn = CrossEntropyLoss()
        self.last_example = -1
        self.last_epoch = -1

        self.embedder:StringEmbedder = None  #  must set in subclasses

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
        if num_edges > conf.max_edges != -1:
            raise TooManyEdges()

        t = time.time()
        x = self.gnn(x, edge_index)

        if conf.print_times:
            print("passed gnn in", (time.time() - t))
        t = time.time()

        # x has now been transformed by the GNN layers. Must map to  a prob dist over candidates
        final_probs = self.pass_output_model(x, example, graph)
        pred_id = torch.argmax(final_probs)
        pred_ans = example.candidates[pred_id]

        if conf.print_times:
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
        if conf.print_times:
            print("got ents in", (time.time() - t))

        x = torch.cat(support_summaries + ent_summaries + candidate_summaries)

        return x

    def pass_output_model(self, x, example, graph, node_id_map=None):
        """
            transformations like pooling can change the effective node ids.
            node_id_map maps the original node ids, to the new effective node ids
        """
        cand_idxs = graph.candidate_nodes
        if node_id_map is not None:
            cand_idxs = [node_id_map[c] for c in cand_idxs]

        cand_embs = x[torch.tensor(cand_idxs).to(device).long(), :]
        cand_probs = self.candidate_scorer(cand_embs).view(len(cand_idxs))

        ent_probs = []
        for c, cand in enumerate(example.candidates):
            """find all entities which match this candidate, and score them each, returning the maximum score"""
            cand_node = graph.ordered_nodes[cand_idxs[c]]
            linked_ent_nodes = set()
            for ent_text in graph.entity_text_to_nodes.keys():  # find all entities with similar text
                if similar(cand_node.text, ent_text):
                    ent_node_ids = graph.entity_text_to_nodes[ent_text]
                    if node_id_map is not None:
                        ent_node_ids = [node_id_map[e] for e in ent_node_ids if e in node_id_map]
                    linked_ent_nodes.update(ent_node_ids)

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
        if pad_volume > conf.max_pad_volume:
            raise PadVolumeOverflow()
        # print("pad vol:", pad_volume)
        query_emb = self.embedder(query)
        support_embeddings = self.coattention.batched_coattention(support_embeddings, query_emb)
        # support_embeddings = [self.coattention(se, query_emb) for se in support_embeddings]
        return support_embeddings


class TooManyEdges(Exception):
    pass


class PadVolumeOverflow(Exception):
    pass