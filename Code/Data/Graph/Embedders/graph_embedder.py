from typing import List, Dict

import torch
from torch import nn
from torch.nn import ModuleDict

from Code.Config import gec, GraphEmbeddingConfig
from Code.Config import graph_construction_config as construction
from Code.Data.Graph.Embedders.graph_encoding import GraphEncoding
from Code.Data.Graph.Embedders.sequence_summariser import SequenceSummariser
from Code.Data.Graph.Embedders.token_embedder import TokenSequenceEmbedder
from Code.Data.Graph.Nodes.span_node import SpanNode
from Code.Data.Graph.context_graph import ContextGraph
from Code.Data.Text.Tokenisation.token_sequence import TokenSequence
from Code.Data.Text.Tokenisation.token_span import TokenSpan
from Code.Data import embedder
from Code.Training import device


class GraphEmbedder(nn.Module):

    """
    Contains all the parameters/functions required to encode the graph nodes
    encodes all nodes, as well as edge features, returns a geometric datapoint
    """

    def __init__(self, gec: GraphEmbeddingConfig):
        super().__init__()
        self.gec: GraphEmbeddingConfig = gec
        self.token_embedder: TokenSequenceEmbedder = embedder
        self.sequence_summarisers: Dict[str, SequenceSummariser] = {}  # maps structure level to a sequence summariser

        self.summarisers: ModuleDict = None

    def on_create_finished(self):
        """called after all summarisers added to register the modules"""
        self.summarisers = ModuleDict(self.sequence_summarisers)

    @staticmethod
    def edge_index(graph: ContextGraph) -> torch.Tensor:
        """
        converts edges into connection info for pytorch geometric
        """
        index = [[], []]  # [[from_ids],[to_ids]]
        for edge in graph.ordered_edges:
            for from_to in range(2):
                index[from_to].append(edge[from_to])
                if not edge.directed:  # adds returning direction
                    index[from_to].append(edge[1-from_to])
        return torch.tensor(index).to(device=device)

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
        context_sequence: TokenSequence = graph.data_sample.context.token_sequence
        embedded_context_sequence: torch.Tensor = self.token_embedder(context_sequence)

        node_features: List[torch.Tensor] = [None] * len(graph.ordered_nodes)

        if self.use_query:
            query_sequence: TokenSequence = graph.query_token_sequence
            embedded_query_sequence: torch.Tensor = self.token_embedder(query_sequence)

        if self.use_query_summary_vec:
            # must map the query embedded sequence into a single embedded summary element
            query_summary: torch.Tensor = self.get_query_vec_summary(embedded_query_sequence)

        for node_id in range(len(graph.ordered_nodes)):
            node = graph.ordered_nodes[node_id]

            if node.get_structure_level() == construction.QUERY_SENTENCE:
                node_features[node_id] = query_summary
                continue

            source_sequence = embedded_context_sequence if node.source == construction.CONTEXT \
                else embedded_query_sequence
            try:
                embedding = self.get_node_embedding(source_sequence, node)
            except Exception as e:
                print("failed to get node embedding for " + repr(node) + " query tok seq: "
                      + repr(graph.query_token_sequence))
                raise e
            node_features[node_id] = embedding
        self.check_dimensions(node_features, graph)
        features = torch.cat(node_features, dim=0).view(len(node_features), -1)

        encoding = GraphEncoding(graph, self.gec, x=features, edge_index=self.edge_index(graph))
        return encoding

    def get_node_embedding(self, full_embedded_sequence: torch.Tensor, node: SpanNode):
        """
        the source of the full embedded sequence should match that of the node.
        however this cannot be verified in this method
        """
        structure_level = node.get_structure_level()
        token_embeddings = self.get_embedded_elements_in_span(full_embedded_sequence, node.token_span)

        if structure_level == construction.TOKEN or structure_level == construction.QUERY_TOKEN:
            """no summariser needed"""
            if token_embeddings.size(1) != 1:
                """token nodes should have exactly 1 seq elem"""
                raise Exception("struc level " + structure_level + " but more than one seq elem in seq "
                                + repr(token_embeddings.size()) + " for node " + repr(node)
                                + "\n full seq:" + repr(full_embedded_sequence.size()))
            return token_embeddings

        embedder = self.sequence_summarisers[structure_level]
        return embedder(token_embeddings)

    def get_query_vec_summary(self, embedded_query_sequence: torch.Tensor):
        return self.sequence_summarisers[construction.QUERY_SENTENCE](embedded_query_sequence)

    @staticmethod
    def get_embedded_elements_in_span(full_embedded_sequence: torch.Tensor, span: TokenSpan):
        start, end = span.subtoken_indexes[0], span.subtoken_indexes[1]
        return full_embedded_sequence[:,start:end,:]

    @staticmethod
    def check_dimensions(node_features: List[torch.Tensor], graph: ContextGraph):
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