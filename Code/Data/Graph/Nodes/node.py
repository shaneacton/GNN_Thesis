from abc import ABC, abstractmethod
from typing import Dict, List

from torch import Tensor


class Node (ABC):

    def __init__(self):
        self.states: Dict[str, Tensor] = {}

    @abstractmethod
    def get_node_states(self) -> Dict[str, Tensor]:
        raise NotImplementedError()

    @abstractmethod
    def get_node_viz_text(self):
        raise NotImplementedError()