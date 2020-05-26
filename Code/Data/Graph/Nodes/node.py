from abc import ABC, abstractmethod
from typing import Dict, List

import torch
from torch import Tensor

from Code.Data.Graph import get_id_from_type
from Code.Data.Graph.State.state_set import StateSet
from Code.Training import device


class Node (ABC):

    TYPE = "type"

    def __init__(self, state_types):
        self.subtype = None
        self.state: StateSet = self.get_state_set(state_types)

    @property
    def states(self) -> Dict[str, Tensor]:
        state = self.state.get_state_tensors()
        state.update({Node.TYPE: self.get_type_tensor()})
        return state

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

    def get_type_tensor(self):
        type_id = get_id_from_type(self.get_type())
        return torch.tensor([type_id]).to(device).squeeze()

    def get_state_set(self, state_types):
        state_set = StateSet()
        for stype in state_types:
            state = stype(self.get_starting_state())
            state_set.add_state(state)
        return state_set

