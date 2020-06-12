from typing import Dict, Set

from torch import Tensor

from Code.Data.Graph.State.state import State


class StateSet(State):

    """
    containr for a named set of state tensor vecs
    belongs to a node or edge
    """

    STATE_SET = "state_set"
    ENTITY_STATE = "entity_state"

    STARTING_STATE = "starting_state"
    CURRENT_STATE = "current_state"
    QUERY_AGNOSTIC_STATE = "query_agnostic_state"

    CHANNEL_STATE = lambda c: "channel("+repr(c)+")"
    TIERED_STATE = lambda t: "tier(" + repr(t) + ")"

    ALL_STATES = [STARTING_STATE, CURRENT_STATE, QUERY_AGNOSTIC_STATE]

    STATE_COMMUNICATION = {
        STARTING_STATE: [],
        CURRENT_STATE: [STARTING_STATE, CURRENT_STATE, QUERY_AGNOSTIC_STATE],
        QUERY_AGNOSTIC_STATE: [QUERY_AGNOSTIC_STATE]
    }

    def __init__(self, name):
        super().__init__(name)
        self.states: Dict[str,StateSet] = {}  # maps state_name to state obj

    def add_state(self, other: State):
        if type(other) == StateSet:
            other:StateSet = other
            self.states.update(other.states)
        else:
            self.states[other.name] = other

    def get_state(self, name):
        return self.states[name]

    def get_named_state_tensors(self):
        named_tensors: Dict[str: Tensor] = {}
        for state in self.states.values():
            # for each contained state object
            named_tensors.update(state.get_named_state_tensors())
        return named_tensors
