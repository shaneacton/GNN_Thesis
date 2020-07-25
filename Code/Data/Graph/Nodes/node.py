from abc import ABC, abstractmethod

from Code.Data.Graph.graph_feature import GraphFeature
from Code.Config import graph_construction_config as construction


class Node (GraphFeature, ABC):

    def __init__(self, source=construction.CONTEXT, subtype=None):
        super().__init__(subtype=subtype)
        self.source = source

    @abstractmethod
    def get_node_viz_text(self):
        raise NotImplementedError()

    @abstractmethod
    def get_structure_level(self):
        raise NotImplementedError()

    def __repr__(self):
        return super(Node, self).__repr__() + " source: " + self.source



