from typing import List, Set

from Code.Data.Graph.Contructors.EdgeExtractors.edge_extractor import EdgeExtractor
from Code.Data.Graph.Edges.edge_relation import EdgeRelation
from Code.Data.Graph.Nodes.node import Node


class FullyConnectedEdgeExtractor(EdgeExtractor):

    def __init__(self, nodes: List[Node], edge_type):
        super().__init__(nodes)
        self.edge_type = edge_type

    def extract_edges(self) -> Set[EdgeRelation]:
        edges: Set[EdgeRelation] = set()
        for f, from_node in enumerate(self.nodes):
            for t, to_node in enumerate(self.nodes):
                if from_node == to_node:
                    continue
                edge_type = self.get_edge_type(from_node, to_node)
                edge = edge_type(f, t)

    def get_edge_type(self, from_node, to_node):
        return self.edge_type