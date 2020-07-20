from torch import nn

from Code.Data.Graph.Embedders.graph_encoding import GraphEncoding
from Code.Data.Graph.context_graph import ContextGraph
from Code.Models import embedder


class GraphEmbedder(nn.Module):

    """
    Contains all the parameters/functions required to encode the graph nodes
    encodes all nodes, as well as edge features, returns a geometric datapoint
    """

    def __init__(self):
        super().__init__()
        self.token_embedder = embedder

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

    def forward(self, graph: ContextGraph) -> GraphEncoding:
        pass