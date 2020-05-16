from abc import ABC, abstractmethod
from typing import List

from Code.Data.Graph.Nodes.node import Node
from Code.Data.Text.context import Context


class NodeExtractor(ABC):

    """
    a node extractor takes in a context and other information and extracts
    a set of typed nodes using a heuristic ruleset
    """

    def __init__(self, context: Context):
        self.context: Context = context

    @abstractmethod
    def extract_nodes(self) -> List[Node]:
        raise NotImplementedError()