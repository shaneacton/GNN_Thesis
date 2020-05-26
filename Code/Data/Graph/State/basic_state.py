from abc import ABC

from torch import Tensor

from Code.Data.Graph.State.state import State


class BasicState(State, ABC):

    def __init__(self, starting_value: Tensor, name):
        super().__init__()
        self.value = starting_value.squeeze()
        self.name = name

    def get_state_tensors(self):
        return {self.name: self.value}