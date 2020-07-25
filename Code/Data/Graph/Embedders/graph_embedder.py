from typing import List, Dict

import torch
from torch import nn

from Code.Config import gec
from Code.Config import graph_construction_config as construction
from Code.Data.Graph.Embedders.graph_encoding import GraphEncoding
from Code.Data.Graph.Embedders.sequence_summariser import SequenceSummariser
from Code.Data.Graph.Embedders.token_embedder import TokenSequenceEmbedder
from Code.Data.Graph.Nodes.span_node import SpanNode
from Code.Data.Graph.context_graph import ContextGraph
from Code.Data.Text.Tokenisation.token_span import TokenSpan
from Code.Models import embedder


class GraphEmbedder(nn.Module):

    """
    Contains all the parameters/functions required to encode the graph nodes
    encodes all nodes, as well as edge features, returns a geometric datapoint
    """

    def __init__(self):
        super().__init__()
        self.token_embedder: TokenSequenceEmbedder = embedder
        self.node_embedders: Dict[str, SequenceSummariser] = {}  # maps structure level to a sequence summariser

    @staticmethod
    def edge_index(graph: ContextGraph):
        """
        converts edges into connection info for pytorch geometric
        """
        index = [[], []]  # [[from_ids],[to_ids]]
        for edge in graph.ordered_edges:
            for from_to in range(2):
                index[from_to].append(edge[from_to])
                if not edge.directed:  # adds returning direction
                    index[from_to].append(edge[1-from_to])
        return index

    @staticmethod
    def edge_types(graph: ContextGraph):
        edge_types = []
        for edge in graph.ordered_edges:
            edge_types.append(edge.get_type_tensor())
            if not edge.directed:  # adds returning directions type
                edge_types.append(edge.get_type_tensor())
        return edge_types

    def use_query(self, graph: ContextGraph):
        has_query_nodes = len(graph.gcc.query_node_types) > 0
        return self.use_query_summary_vec or has_query_nodes

    def use_query_summary_vec(self, graph: ContextGraph) -> bool:
        query_sentence_node = construction.QUERY_SENTENCE in graph.gcc.query_node_types
        return query_sentence_node or gec.use_query_aware_context_vectors

    def forward(self, graph: ContextGraph) -> GraphEncoding:
        context_sequence = graph.data_sample.context.token_sequence
        embedded_context_sequence = self.token_embedder(context_sequence)

        node_features: List[torch.Tensor] = [None] * len(graph.ordered_edges)

        if self.use_query:
            query_sequence = graph.query_token_sequence
            embedded_query_sequence = self.token_embedder(query_sequence)

        if self.use_query_summary_vec:
            # must map the query embedded sequence into a single embedded summary element
            query_summary = self.get_query_vec_summary(embedded_query_sequence)

        for node_id in range(len(graph.ordered_edges)):
            node = graph.ordered_nodes[node_id]

            if node.get_structure_level() == construction.QUERY_SENTENCE:
                node_features[node_id] = query_summary
                continue

            source_sequence = embedded_context_sequence if node.source else query_sequence
            embedding = self.get_node_embedding(source_sequence, node)
            node_features[node_id] = embedding

    def get_node_embedding(self, full_embedded_sequence: List[torch.Tensor], node: SpanNode):
        """
        the source of the full embedded sequence should match that of the node.
        however this cannot be verified in this method
        """
        structure_level = node.get_structure_level()
        embedder = self.node_embedders[structure_level]

        node: SpanNode = node
        token_embeddings = self.get_embedded_elements_in_span(full_embedded_sequence, node.token_span)
        return embedder(token_embeddings)

    def get_query_vec_summary(self, embedded_query_sequence: List[torch.Tensor]):
        pass

    @staticmethod
    def get_embedded_elements_in_span(full_embedded_sequence: List[torch.Tensor], span: TokenSpan):
        return full_embedded_sequence[span.subtoken_indexes[0]:span.subtoken_indexes[1]]