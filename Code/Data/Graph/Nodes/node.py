from abc import ABC, abstractmethod
from typing import Dict

from torch import Tensor

from Code.Data.Graph.graph_feature import GraphFeature


class Node (GraphFeature, ABC):

    TYPE = "type"

    def __init__(self, subtype=None):
        super().__init__(subtype=subtype)

    @property
    def states_tensors(self) -> Dict[str, Tensor]:
        state = {Node.TYPE: self.get_type_tensor()}
        state.update(self.get_all_node_state_tensors())
        return state

    @abstractmethod
    def get_node_viz_text(self):
        raise NotImplementedError()

    def get_all_node_state_tensors(self) -> Dict[str, Tensor]:
        return {}




