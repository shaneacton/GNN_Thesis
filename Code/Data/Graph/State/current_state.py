from Code.Data.Graph.State.basic_state import BasicState
from Code.Data.Graph.State.state_set import StateSet


class CurrentState(BasicState):

    def __init__(self, starting_state):
        super().__init__(starting_state, StateSet.CURRENT_STATE)
