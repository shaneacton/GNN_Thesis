import inspect
import time
from typing import List, Tuple

import torch
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss, ReLU

from Code.Embedding.gru_contextualiser import GRUContextualiser
from Code.Embedding.string_embedder import StringEmbedder
from Code.GNNs.gated_gnn import GatedGNN
from Code.GNNs.gnn_stack import GNNStack
from Code.HDE.Graph.graph import HDEGraph
from Code.Transformers.coattention import Coattention
from Code.Transformers.summariser import Summariser
from Code.Transformers.switch_summariser import SwitchSummariser
from Code.HDE.scorer import HDEScorer
from Code.HDE.wikipoint import Wikipoint
from Code.Training import dev
from Code.Utils.graph_utils import get_entity_summaries, similar
from Code.constants import DOCUMENT, CANDIDATE
from Config.config import conf


class HDEModel(nn.Module):

    def __init__(self, GNN_CLASS=None, **kwargs):
        from Code.Utils.model_utils import num_params

        if GNN_CLASS is None:
            from Code.Utils.model_utils import GNN_MAP
            GNN_CLASS = GNN_MAP[conf.gnn_class]
        super().__init__()
        self.name = conf.model_name
        self.hidden_size = conf.hidden_size
        self.use_gating = conf.use_gating

        self.coattention = Coattention(**kwargs)
        conf.cfg["num_coattention_params"] = num_params(self.coattention)

        if conf.use_gru_contextualiser:
            self.supp_contextualiser = GRUContextualiser()
            self.cand_contextualiser = GRUContextualiser()
            self.query_contextualiser = GRUContextualiser()

        if conf.use_switch_summariser:
            self.summariser = SwitchSummariser(**kwargs)
        else:
            self.summariser = Summariser(**kwargs)
        conf.cfg["num_summariser_params"] = num_params(self.summariser)
        conf.cfg["num_transformer_params"] = num_params(self.summariser) + num_params(self.coattention)

        self.relu = ReLU()

        self.gnn = None
        self.init_gnn(GNN_CLASS)
        if conf.use_big_gating:
            """wraps the whole gnn in a gate, as in Hierarchical Graph Network"""
            self.gnn = GatedGNN(self.gnn)
        conf.cfg["num_gnn_params"] = num_params(self.gnn)

        self.candidate_scorer = HDEScorer(conf.hidden_size)
        self.entity_scorer = HDEScorer(conf.hidden_size)

        conf.cfg["num_output_params"] = num_params(self.candidate_scorer) + num_params(self.entity_scorer)

        self.loss_fn = CrossEntropyLoss()
        self.last_example = -1
        self.last_epoch = -1

        self.embedder:StringEmbedder = None  #  must set in subclasses
        conf.cfg["num_total_params"] = num_params(self)

    def init_gnn(self, GNN_CLASS):
        init_args = inspect.getfullargspec(GNN_CLASS.__init__)[0]
        if "heads" in init_args:
            args = {"heads": conf.heads}
        else:
            args = {}
        self.gnn = GNNStack(GNN_CLASS, **args)


    def forward(self, example: Wikipoint=None, graph: HDEGraph=None):
        """
            nodes are created for each support, as well as each candidate and each context entity
            nodes are concattenated as follows: supports, entities, candidates

            nodes are connected according to the HDE paper
            the graph is converted to a pytorch geometric datapoint
        """
        if graph is None:
            graph = self.create_graph(example)  # heuristic
        else:
            example = graph.example
        x = self.get_graph_features(example)  # learned

        num_edges = len(graph.unique_edges)
        if num_edges > conf.max_edges != -1:
            raise TooManyEdges()
        if conf.show_memory_usage_data:
            print("num edges:", num_edges)

        x = self.pass_gnn(x, example, graph)
        if isinstance(x, Tuple):
            args = x[1:]
            x = x[0]
        else:
            args = []
        # x has now been transformed by the GNN layers. Must map to  a prob dist over candidates
        return self.finish(x, example, graph, *args)

    def pass_gnn(self, x, example, graph):
        edge_index = graph.edge_index()
        t = time.time()
        x = self.gnn(x, edge_index=edge_index)
        if conf.print_times:
            print("passed gnn in", (time.time() - t))
        return x

    def finish(self, x, example, graph, *args):
        final_probs = self.pass_output_model(x, example, graph, *args)
        pred_id = torch.argmax(final_probs)
        pred_ans = example.candidates[pred_id]

        if example.answer is not None:
            ans_id = example.candidates.index(example.answer)
            probs = final_probs.view(1, -1)  # batch dim
            ans = torch.tensor([ans_id]).to(dev())
            loss = self.loss_fn(probs, ans)

            return loss, pred_ans

        return pred_ans

    def get_graph_features(self, example):
        """
            performs coattention between the query and context sequence
            then summarises subsequences of tokens according to node spans
            yielding the same-sized node features
        """
        t = time.time()
        support_embeddings = [self.embedder(sup) for sup in example.supports]
        if conf.use_gru_contextualiser:
            support_embeddings = [self.supp_contextualiser(sup) for sup in support_embeddings]

        query_aware_support_embeddings = self.get_query_aware_context_embeddings(support_embeddings, example.query)
        if conf.print_times:
            print("got supp embs in", (time.time() - t))
        cand_embs = [self.embedder(cand) for cand in example.candidates]
        if conf.use_gru_contextualiser:
            cand_embs = [self.cand_contextualiser(cand) for cand in cand_embs]

        candidate_summaries = self.summariser(cand_embs, CANDIDATE)
        support_summaries = self.summariser(query_aware_support_embeddings, DOCUMENT)

        t = time.time()

        ent_summaries = get_entity_summaries(example.ent_token_spans, query_aware_support_embeddings, self.summariser)
        if conf.print_times:
            print("got ent summaries in", (time.time() - t))

        x = torch.cat(support_summaries + ent_summaries + candidate_summaries)

        return x

    def get_query_aware_context_embeddings(self, support_embeddings: List[Tensor], query: str):
        """uses the coattention module to bring info from the query into the context"""
        # print("supps:", [s.size() for s in support_embeddings])
        pad_volume = max([s.size(1) for s in support_embeddings]) * len(support_embeddings)
        if pad_volume > conf.max_pad_volume:
            raise PadVolumeOverflow()
        if conf.show_memory_usage_data:
            print("documents padded volume:", pad_volume)
        # print("pad vol:", pad_volume)
        query_emb = self.embedder(query, allow_unknowns=False)
        if conf.use_gru_contextualiser:
            query_emb = self.query_contextualiser(query_emb)

        support_embeddings = self.coattention.batched_coattention(support_embeddings, query_emb)
        return support_embeddings

    def pass_output_model(self, x, example, graph, node_id_map=None):
        """
            transformations like pooling can change the effective node ids.
            node_id_map maps the original node ids, to the new effective node ids
        """
        cand_idxs = graph.candidate_nodes
        if node_id_map is not None:
            cand_idxs = [node_id_map[c] for c in cand_idxs]

        cand_embs = x[torch.tensor(cand_idxs).to(dev()).long(), :]
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
                max_prob = torch.tensor(0.).to(dev())
            else:
                linked_ent_nodes = sorted(list(linked_ent_nodes))  # node ids of all entities linked to this candidate
                linked_ent_nodes = torch.tensor(linked_ent_nodes).to(dev()).long()
                ent_embs = torch.index_select(x, dim=0, index=linked_ent_nodes)

                cand_ent_probs = self.entity_scorer(ent_embs)
                max_prob = torch.max(cand_ent_probs)
            ent_probs += [max_prob]
        ent_probs = torch.stack(ent_probs, dim=0)

        final_probs = cand_probs + ent_probs
        return final_probs



class TooManyEdges(Exception):
    pass


class PadVolumeOverflow(Exception):
    pass