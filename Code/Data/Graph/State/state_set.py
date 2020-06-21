from typing import Dict

from torch import Tensor

from Code.Data.Graph.Nodes.span_node import SpanNode
from Code.Data.Graph.State.basic_state import BasicState
from Code.Data.Graph.State.state import State


class StateSet(State):

    """
    containr for a named set of state tensor vecs
    belongs to a node or edge
    """

    CONTEXT = "context"
    QUERY = "query"

    STATE_SET = "state_set"

    STARTING_STATE = "starting_state"
    PREVIOUS_STATE = "previous_state"
    CURRENT_STATE = "current_state"

    QUERY_AGNOSTIC_STATE = "query_agnostic_state"

    CHANNEL_STATE_NAME = lambda c: "channel(" + repr(c) + ")"
    TIERED_STATE_NAME = lambda t: "tier(" + repr(t) + ")"


    STATE_COMMUNICATION = {
        STARTING_STATE: [],
        CURRENT_STATE: [STARTING_STATE, PREVIOUS_STATE, CURRENT_STATE, QUERY_AGNOSTIC_STATE],
        QUERY_AGNOSTIC_STATE: [QUERY_AGNOSTIC_STATE]
    }

    CONFIG = {
        "use_starting_state": True,
        "use_previous_state": False
    }

    def __init__(self, name):
        super().__init__(name)
        self.states: Dict[str,State] = {}  # maps state_name to state obj

    @property
    def current_state(self):
        return self.states[StateSet.CURRENT_STATE]

    @property
    def previous_state(self):
        return self.states[StateSet.PREVIOUS_STATE]

    @property
    def starting_state(self):
        return self.states[StateSet.STARTING_STATE]

    @property
    def query_state(self):
        return self.states[StateSet.QUERY]

    def add_state(self, other: State):
        if type(other) == StateSet:
            other:StateSet = other
            self.states.update(other.states)
        else:
            self.states[other.name] = other

    def get_state(self, name) -> State:
        return self.states[name]

    def get_named_state_tensors(self):
        named_tensors: Dict[str: Tensor] = {}
        for state in self.states.values():
            # for each contained state object
            named_tensors.update(state.get_named_state_tensors())
        return named_tensors

    def add_states_from_named_vecs(self, named_state_vecs: Dict[str, Tensor]):
        for name, vec in named_state_vecs.items():
            state = BasicState(vec, name)
            self.add_state(state)

    def initialise_states(self):
        """
        converts the initially present state eg token_ids into this statesets current, previous and starting states
        """

        if StateSet.STARTING_STATE in self.states:
            start = self.states[StateSet.STARTING_STATE]
        elif SpanNode.EMB_IDS in self.states:
            emb_ids = self.states[SpanNode.EMB_IDS]
            #convert to embeddings and pad
            #add start state
            raise Exception()
        else:
            raise Exception()

        starting_vec = start.

        if StateSet.CONFIG["use_starting_state"]:
            self.add_state(BasicState(start., StateSet.STARTING_STATE))
        if StateSet.CONFIG["use_previous_state"]:
            self.add_state(BasicState(start_vec, StateSet.PREVIOUS_STATE))


