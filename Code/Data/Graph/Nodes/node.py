from abc import ABC, abstractmethod

import Code.constants
from Code.Data.Graph.graph_feature import GraphFeature


class Node (GraphFeature, ABC):

    def __init__(self, source=Code.constants.CONTEXT, subtype=None):
        super().__init__(subtype=subtype)
        self.source = source

    @abstractmethod
    def get_node_viz_text(self):
        raise NotImplementedError()

    def __repr__(self):
        return super(Node, self).__repr__() + " source: " + self.source



