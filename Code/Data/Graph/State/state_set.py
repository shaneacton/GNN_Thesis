from typing import Dict, Set

from torch import Tensor

from Code.Data.Graph.State.basic_state import BasicState
from Code.Data.Graph.State.state import State
from Code.Data.Graph.State.tiered_state import TieredState


class StateSet(State):

    """
    containr for a named set of state tensor vecs
    belongs to a node or edge
    """

    ENTITY_STATE = "entity_state"


    STARTING_STATE = "starting_state"
    CURRENT_STATE = "current_state"
    QUERY_AGNOSTIC_STATE = "query_agnostic_state"

    CHANNEL_STATE = lambda c: "channel("+repr(c)+")"
    TIERED_STATE = lambda t: "tier(" + repr(t) + ")"

    ALL_STATES = [STARTING_STATE, CURRENT_STATE, QUERY_AGNOSTIC_STATE]


    STATE_TYPE_MAP = {STARTING_STATE: BasicState, CURRENT_STATE: BasicState,
                      QUERY_AGNOSTIC_STATE: (TieredState, {"num_channels": 2})}

    STATE_COMMUNICATION = {
        STARTING_STATE: [],
        CURRENT_STATE: [STARTING_STATE, CURRENT_STATE, QUERY_AGNOSTIC_STATE],
        QUERY_AGNOSTIC_STATE: [QUERY_AGNOSTIC_STATE]
    }

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
