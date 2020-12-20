import time
from typing import List, Dict, Callable, Tuple, Union

import torch
from torch import nn, Tensor
from torch.nn import ModuleDict
from transformers import PreTrainedTokenizerFast, TokenSpan, BatchEncoding

from Code.Config import GraphEmbeddingConfig, GraphConstructionConfig
from Code.Data.Graph.Contructors.qa_graph_constructor import QAGraphConstructor
from Code.Data.Graph.Embedders.Summarisers.sequence_summariser import SequenceSummariser
from Code.Data.Graph.graph_encoding import GraphEncoding
from Code.Data.Graph.Nodes.span_node import SpanNode
from Code.Data.Graph.Types.type_map import TypeMap
from Code.Data.Graph.Types.types import Types
from Code.Data.Graph.context_graph import QAGraph
from Code.Data.Text.longformer_embedder import LongformerEmbedder
from Code.Data.Text.text_utils import question, context, is_batched, has_candidates
from Code.Training.Utils.initialiser import get_tokenizer
from Code.Training.Utils.text_encoder import TextEncoder
from Code.Play.examples import test_example
from Code.Training import device
from Code.Training.Utils.metric import Metric
from Code.constants import TOKEN, CONTEXT, QUERY


class GraphEmbedder(nn.Module):

    """
    Contains all the parameters/functions required to encode the graph nodes
    encodes all nodes, as well as edge features, returns a geometric datapoint
    """

    def __init__(self, gec: GraphEmbeddingConfig, tokeniser: PreTrainedTokenizerFast=None,
                 long_embedder: Callable[[BatchEncoding], Tensor]=None, num_features=-1, gcc=None):
        """
        :param long_embedder: any function which maps a batchencoding to a tensor of features
        :param num_features: if -1 will default to dims of pretrained embedder
        """
        super().__init__()
        self.gcc: GraphConstructionConfig = gcc
        if not long_embedder:
            long_embedder = LongformerEmbedder(out_features=num_features)
        if not tokeniser:
            tokeniser = get_tokenizer()
        self.long_embedder = long_embedder
        self.text_encoder = TextEncoder(tokeniser)

        self.gec: GraphEmbeddingConfig = gec
        # maps source, structure level to a sequence summariser instance
        self.sequence_summarisers: Dict[str, Dict[str, SequenceSummariser]] = {}

        self.context_summarisers: ModuleDict = None  # used later to register summariser params
        self.query_summarisers: ModuleDict = None  # used later to register summariser params

        self.node_type_map = TypeMap()  # maps node types to ids
        self.edge_type_map = TypeMap()  # maps edge types to ids

        self.embedding_times = Metric("embedding times")

    def on_create_finished(self):
        """called after all summarisers added to register the modules"""
        if self.gcc.structure_levels[CONTEXT] != [TOKEN]:
            # no summarisers for tokens
            self.context_summarisers = ModuleDict(self.sequence_summarisers[CONTEXT])
        if self.gcc.structure_levels[QUERY] != [TOKEN]:
            self.query_summarisers = ModuleDict(self.sequence_summarisers[QUERY])

    @staticmethod
    def edge_index(graph: QAGraph) -> torch.Tensor:
        """
        converts edges into connection info for pytorch geometric
        """
        index = [[], []]  # [[from_ids],[to_ids]]
        for edge in graph.ordered_edges:
            for from_to in range(2):
                index[from_to].append(edge[from_to])
                if not edge.directed:  # adds returning direction
                    index[from_to].append(edge[1-from_to])
        return torch.tensor(index).to(device)

    def edge_types(self, graph: QAGraph) -> torch.Tensor:
        edge_types = []
        for edge in graph.ordered_edges:
            type_id = self.edge_type_map.get_id(edge)
            edge_types.append(type_id)
            if not edge.directed:  # adds returning directions type
                edge_types.append(type_id)
        return torch.tensor(edge_types).to(device)

    def node_types(self, graph: QAGraph) -> torch.Tensor:
        node_types = []
        for node in graph.ordered_nodes:
            type_id = self.node_type_map.get_id(node)
            node_types.append(type_id)
        return torch.tensor(node_types).to(device)

    def get_source_spans(self, qa_encoding: BatchEncoding, example: Dict) \
            -> Tuple[TokenSpan, TokenSpan, Union[TokenSpan, None]]:
        """
        qa enc is <context><query>[candidates_string]
        :return: ctx_span, q_span_ optional[cands_span]
        these token spans respect the <s>,<\\s> tokens, and should be faithful to the token order in qa_enc
        """
        if is_batched(example):
            raise Exception()
        ctx_encoding = self.text_encoder.tokeniser.encode_plus(context(example))
        qu_encoding = self.text_encoder.tokeniser.encode_plus(question(example))
        ctx_len = len(ctx_encoding['input_ids'])
        qu_len = len(qu_encoding['input_ids'])
        full_len = len(qa_encoding['input_ids'])

        context_span = TokenSpan(1, ctx_len - 1)
        query_span = TokenSpan(ctx_len + 1, ctx_len + qu_len - 1)

        cands_span = None

        if has_candidates(example):
            cands_span = TokenSpan(ctx_len + qu_len - 1, full_len - 1)
            if ctx_len + qu_len + cands_span.end - cands_span.start != full_len:
                raise Exception("context len: " + repr(ctx_len) + " query len: " + repr(qu_len) + " cands len: " + repr(cands_span.end - cands_span.start) + " full len:", full_len)
        elif ctx_len + qu_len != full_len:
            raise Exception("context len: " + repr(ctx_len) + " query len: " + repr(qu_len) + " full len:", full_len)

        return context_span, query_span, cands_span

    def forward(self, graph: QAGraph) -> GraphEncoding:
        if isinstance(graph, List):
            raise Exception("single graph embedder cannot handle batched graphs: " + repr(graph))
        # print("running graph embedder on context:", context_sequence)
        start_time = time.time()
        qa_encoding = self.text_encoder.get_encoding(graph.example)
        context_span, query_span, cands_span = self.get_source_spans(qa_encoding, graph.example)
        full_embedded_sequence = self.long_embedder(qa_encoding)
        embedded_context_sequence: torch.Tensor = self.get_embedded_elements_in_span(full_embedded_sequence, context_span)
        embedded_query_sequence: torch.Tensor = self.get_embedded_elements_in_span(full_embedded_sequence, query_span)
        if cands_span:
            embedded_cands_sequence: torch.Tensor = self.get_embedded_elements_in_span(full_embedded_sequence,
                                                                                       cands_span)

        node_features: List[torch.Tensor] = [None] * len(graph.ordered_nodes)
        for node_id in range(len(graph.ordered_nodes)):
            node = graph.ordered_nodes[node_id]
            source_sequence = embedded_context_sequence if node.source == CONTEXT else \
                (embedded_query_sequence if node.source == QUERY else embedded_cands_sequence)

            try:
                embedding = self.get_node_embedding(source_sequence, node)
                node_features[node_id] = embedding
            except Exception as e:
                print("failed to get node embedding for " + repr(node))
                raise e

        self.check_dimensions(node_features, graph)
        features = torch.cat(node_features, dim=0).view(len(node_features), -1)

        node_types = self.node_types(graph)
        edge_types = self.edge_types(graph)
        types = Types(self.node_type_map, self.edge_type_map, node_types, edge_types)

        encoding = GraphEncoding(graph, self.gec, types, x=features, edge_index=self.edge_index(graph))

        self.embedding_times.report(time.time() - start_time)
        return encoding

    def get_node_embedding(self, full_embedded_sequence: torch.Tensor, node: SpanNode):
        """
        the source of the full embedded sequence should match that of the node.
        however this cannot be verified in this method
        """
        structure_level = node.get_structure_level()
        token_embeddings = self.get_embedded_elements_in_span(full_embedded_sequence, node.token_span)

        if structure_level == TOKEN:
            """no summariser needed"""
            if token_embeddings.size(1) != 1:
                """token nodes should have exactly 1 seq elem"""
                raise Exception("struc level " + structure_level + " but more than one seq elem in seq "
                                + repr(token_embeddings.size()) + " for node " + repr(node)
                                + "\n full seq:" + repr(full_embedded_sequence.size()))
            return token_embeddings

        if node.source not in self.sequence_summarisers:
            raise Exception(repr(node) + " source not in " + repr(list(self.sequence_summarisers.keys())))
        if structure_level not in self.sequence_summarisers[node.source]:
            raise Exception(repr(node) + " structure lev " + structure_level +" not in " +
                            repr(list(self.sequence_summarisers[node.source].keys())))

        summ = self.sequence_summarisers[node.source][structure_level]
        return summ(token_embeddings)

    @staticmethod
    def get_embedded_elements_in_span(full_embedded_sequence: torch.Tensor, span: TokenSpan):
        """cuts the relevant elements out to return the spans elements only"""
        embs = full_embedded_sequence[:, span.start:span.end, :]
        if embs.size(1) == 0:
            raise Exception("cannot get emb elements from span: " + repr(span) + " given seq: " + repr(full_embedded_sequence.size()))
        return embs

    @staticmethod
    def check_dimensions(node_features: List[torch.Tensor], graph: QAGraph):
        size = -1
        for i in range(len(node_features)):
            node = graph.ordered_nodes[i]
            if node_features[i] is None:
                raise Exception("no node feature created for node: " + repr(node))
            if not isinstance(node_features[i], torch.Tensor):
                raise Exception("non tensor ("+repr(node_features[i])+") feature created for node: " + repr(node))

            node_size = node_features[i].size()
            if size == -1:
                size = node_size
                continue
            if size != node_size:
                first_node = graph.ordered_nodes[0]
                raise Exception("size mismatch. Got " + repr(size) + " for " + repr(first_node) + " but " + repr(node_size) + " for " + repr(node))


if __name__ == "__main__":
    from Code.Config import gec
    from Code.Config import gcc

    embedder = gec.get_graph_embedder(gcc)

    const = QAGraphConstructor(gcc)

    print(test_example)

    graph = const._create_single_graph_from_data_sample(test_example)
    encoding = embedder(graph)
    print("done")