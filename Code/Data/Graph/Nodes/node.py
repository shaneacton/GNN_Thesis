from abc import ABC, abstractmethod
from typing import Dict, List

from torch import Tensor

from Code.Data.Graph.State.state_set import StateSet


class Node (ABC):

    def __init__(self, state_types):
        self.subtype = None
        self.state: StateSet = self.get_state_set(state_types)

    @property
    def states(self) -> Dict[str, Tensor]:
        return self.state.get_state_tensors()

    @abstractmethod
    def get_node_viz_text(self):
        raise NotImplementedError()

    @abstractmethod
    def get_starting_state(self) -> Tensor:
        raise NotImplementedError()

    def get_type(self):
        """
        returns type, subtype
        only some nodes have subtypes
        """
        return type(self), self.subtype

    def get_state_set(self, state_types):
        state_set = StateSet()
        for stype in state_types:
            state = stype(self.get_starting_state())
            state_set.add_state(state)
        return state_set

