from abc import ABC

from Code.Data.Graph.State.state import State


class BasicState(State, ABC):

    def __init__(self, starting_value, name):
        super().__init__()
        self.value=starting_value
        self.name=name

    def get_state_tensors(self):
        return {self.name: self.value}