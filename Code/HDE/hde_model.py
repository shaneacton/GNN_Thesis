import inspect
import time
from typing import List, Tuple

import torch
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss, ReLU

from Code.Embedding.bert_embedder import BertEmbedder
from Code.Embedding.glove_embedder import GloveEmbedder
from Code.Embedding.gru_contextualiser import GRUContextualiser
from Code.Embedding.string_embedder import StringEmbedder
from Code.GNNs.gnn_stack import GNNStack
from Code.GNNs.transformer_gnn import TransformerGNN
from Code.HDE.Graph.graph import HDEGraph
from Code.HDE.scorer import HDEScorer
from Code.Training import dev
from Code.Training.timer import log_time
from Code.Transformers.summariser import Summariser
from Code.Transformers.switch_summariser import SwitchSummariser
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
        self.use_gating = conf.use_gating

        if not conf.use_simple_hde:
            self.supp_contextualiser = GRUContextualiser()
            self.cand_contextualiser = GRUContextualiser()
            self.query_contextualiser = GRUContextualiser()

        if conf.use_switch_summariser:
            self.summariser = SwitchSummariser(**kwargs)
        else:
            self.summariser = Summariser(**kwargs)
        conf.cfg["num_summariser_params"] = num_params(self.summariser)

        self.relu = ReLU()

        self.gnn = None
        self.init_gnn(GNN_CLASS)
        conf.cfg["num_gnn_params"] = num_params(self.gnn)

        self.candidate_scorer = HDEScorer(conf.hidden_size)
        self.entity_scorer = HDEScorer(conf.hidden_size)

        conf.cfg["num_output_params"] = num_params(self.candidate_scorer) + num_params(self.entity_scorer)

        self.loss_fn = CrossEntropyLoss()
        self.last_example = -1
        self.last_epoch = -1

        if conf.embedder_type == "bert":
            self.embedder: StringEmbedder = BertEmbedder()
        elif conf.embedder_type == "glove":
            self.embedder: StringEmbedder = GloveEmbedder()
        else:
            raise Exception("unreckognised embedder type: " + repr(conf.embedder_type) + " needs: {bert, glove}")
        conf.cfg["num_embedding_params"] = num_params(self.embedder)
        conf.cfg["num_total_params"] = num_params(self)

    def init_gnn(self, GNN_CLASS):
        init_args = inspect.getfullargspec(GNN_CLASS.__init__)[0]
        if "heads" in init_args:
            args = {"heads": conf.heads}
        else:
            args = {}
        if hasattr(conf, "use_transformer_gnn") and conf.use_transformer_gnn:
            self.gnn = TransformerGNN(heads=conf.heads, num_layers=conf.num_layers, **args)
        else:
            self.gnn = GNNStack(GNN_CLASS, **args)

    def forward(self, graph: HDEGraph):
        """
            nodes are created for each support, as well as each candidate and each context entity
            nodes are concattenated as follows: supports, entities, candidates

            nodes are connected according to the HDE paper
            the graph is converted to a pytorch geometric datapoint
        """

        x = self.get_graph_features(graph.example)  # learned
        assert x.size(0) == len(graph.ordered_nodes), "error in feature extraction. num node features: " + repr(x.size(0)) + " num nodes: " + repr(len(graph.ordered_nodes))

        num_edges = len(graph.unique_edges)
        if num_edges > conf.max_edges != -1:
            raise TooManyEdges()
        if conf.show_memory_usage_data:
            print("num edges:", num_edges)

        x = self.pass_gnn(x, graph)
        if isinstance(x, Tuple):
            args = x[1:]
            x = x[0]
        else:
            args = []
        # x has now been transformed by the GNN layers. Must map to  a prob dist over candidates
        return self.finish(x, graph, *args)

    def pass_gnn(self, x, graph):
        t = time.time()
        kwargs = {}
        if conf.use_switch_gnn:
            kwargs["graph"] = graph
        if hasattr(conf, "use_transformer_gnn") and conf.use_transformer_gnn:
            # print("x before:", x.size(), x)
            x = self.gnn(x, graph.get_mask()).view(x.size(0), -1, **kwargs)
            # print("x after:", x.size(), x)
        else:
            x = self.gnn(x, edge_index=graph.edge_index(), **kwargs)
        log_time("gnn", time.time() - t)
        return x

    def finish(self, x, graph, *args):
        t = time.time()
        final_probs = self.pass_output_model(x, graph, *args)
        pred_id = torch.argmax(final_probs)
        pred_ans = graph.example.candidates[pred_id]

        if graph.example.answer is not None:
            ans_id = graph.example.candidates.index(graph.example.answer)
            probs = final_probs.view(1, -1)  # batch dim
            ans = torch.tensor([ans_id]).to(dev())
            loss = self.loss_fn(probs, ans)
            log_time("output", time.time() - t)
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
        query_emb = self.embedder(example.query, allow_unknowns=False)
        cand_embs = [self.embedder(cand) for cand in example.candidates]
        self.check_pad_volume(support_embeddings)

        if not conf.use_simple_hde:
            gru_t = time.time()
            support_embeddings = [self.supp_contextualiser(sup) for sup in support_embeddings]
            query_emb = self.query_contextualiser(query_emb)
            cand_embs = [self.cand_contextualiser(cand_emb) for cand_emb in cand_embs]
            log_time("GRUs", time.time()-gru_t, increment_counter=False)  # signals end of example, needed multiple calls

        node_t = time.time()
        candidate_summaries = self.summariser(cand_embs, CANDIDATE, query_vec=query_emb)
        support_summaries = self.summariser(support_embeddings, DOCUMENT, query_vec=query_emb)

        ent_summaries = get_entity_summaries(example.ent_token_spans, support_embeddings, self.summariser, query_vec=query_emb)
        x = torch.cat(support_summaries + ent_summaries + candidate_summaries)
        log_time("Graph embedding", time.time() - t)
        log_time("Node embedding", time.time() - node_t)
        log_time("Token embedding", 0, increment_counter=True)  # signals end of example, needed multiple calls
        log_time("GRUs", 0, increment_counter=True)  # signals end of example, needed multiple calls

        return x

    def check_pad_volume(self, support_embeddings: List[Tensor]):
        """
            uses the coattention module to bring info from the query into the context
            if return_query_encoding=False, the encoded sequences will be cropped to return to their pre-concat size
        """
        pad_volume = max([s.size(1) for s in support_embeddings]) * len(support_embeddings)
        if pad_volume > conf.max_pad_volume:
            raise PadVolumeOverflow()
        if conf.show_memory_usage_data:
            print("documents padded volume:", pad_volume)

        return support_embeddings

    def pass_output_model(self, x, graph, node_id_map=None):
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
        for c, cand in enumerate(graph.example.candidates):
            """find all entities which match this candidate, and score them each, returning the maximum score"""
            cand_node = graph.ordered_nodes[cand_idxs[c]]
            linked_ent_nodes = set()
            for ent_text in graph.entity_text_to_nodes.keys():  # find all entities with similar text
                if similar(cand_node.text, ent_text):
                    ent_node_ids = graph.entity_text_to_nodes[ent_text]
                    if node_id_map is not None:
                        ent_node_ids = [node_id_map[e] for e in ent_node_ids if e in node_id_map]
                    linked_ent_nodes.update(ent_node_ids)

            if len(linked_ent_nodes) == 0:  # no mentions of this candidate
                ent_prob = torch.tensor(0.).to(dev())
            else:
                linked_ent_nodes = sorted(list(linked_ent_nodes))  # node ids of all entities linked to this candidate
                linked_ent_nodes = torch.tensor(linked_ent_nodes).to(dev()).long()
                ent_embs = torch.index_select(x, dim=0, index=linked_ent_nodes)

                cand_ent_probs = self.entity_scorer(ent_embs)

                if conf.use_average_output_agg:
                    ent_prob = torch.sum(cand_ent_probs)  # todo - divide by num cands
                else:
                    ent_prob = torch.max(cand_ent_probs)
            ent_probs += [ent_prob]
        ent_probs = torch.stack(ent_probs, dim=0)

        final_probs = cand_probs + ent_probs
        return final_probs


class TooManyEdges(Exception):
    pass


class PadVolumeOverflow(Exception):
    pass