from typing import Dict, Set

from torch import Tensor

from Code.Data.Graph.State.state import State


class StateSet(State):

    """
    containr for a named set of state tensor vecs
    belongs to a node or edge
    """

    STARTING_STATE = "starting_state"
    CURRENT_STATE = "current_state"
    TIERED_STATE_PREFIX = "tier"

    def __init__(self):
        self.states: Set[StateSet] = set()

    def add_state(self, other: State):
        if type(other) == StateSet:
            other:StateSet = other
            self.states += other.states
        else:
            self.states.add(other)

    def get_state_tensors(self):
        named_tensors: Dict[str: Tensor] = {}
        for state in self.states:
            # for each contained state object
            named_tensors.update(state.get_state_tensors())
        return named_tensors
