from abc import ABC, abstractmethod
from typing import List, Set

from Code.Data.Graph.Edges.edge_relation import EdgeRelation
from Code.Data.Graph.Nodes.node import Node


class EdgeExtractor(ABC):

    """
    an edge extractor takes in a set of nodes, and uses a heuristic rule to
    create and return a set of typed edges
    """

    def __init__(self, nodes: List[Node]):
        self.nodes: List[Node] = nodes

    @abstractmethod
    def extract_edges(self) -> Set[EdgeRelation]:
        raise NotImplementedError()

    @abstractmethod
    def get_edge_type(self, from_node, to_node):
        raise NotImplementedError()
