from abc import ABC, abstractmethod
from typing import Dict

from torch import Tensor

from Code.Data.Graph.graph_feature import GraphFeature


class Node (GraphFeature, ABC):

    TYPE = "type"

    def __init__(self, subtype=None):
        super().__init__(subtype=subtype)

    @abstractmethod
    def get_node_viz_text(self):
        raise NotImplementedError()




