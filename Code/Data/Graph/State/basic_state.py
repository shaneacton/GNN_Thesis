from abc import ABC

from torch import Tensor

from Code.Data.Graph.State.state import State


class BasicState(State, ABC):

    def __init__(self, starting_value: Tensor, name, type=None):
        type_arg = {"type":type} if type else {}
        super().__init__(name, **type_arg)
        self.value = starting_value.squeeze()

    def get_named_state_tensors(self):
        return {self.name: self.value}