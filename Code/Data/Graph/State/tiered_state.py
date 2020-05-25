from typing import Dict

from torch import Tensor

from Code.Data.Graph.State.state import State
from Code.Data.Graph.State.state_set import StateSet


class TieredState(State):
    """
    simply an array of states which are converted into separate named states
    """

    def __init__(self, starting_state, num_tiers=3):
        super().__init__()
        self.states=[starting_state]*num_tiers

    def get_state_tensors(self):
        named_states: Dict[str: Tensor] = {}
        for i, state in enumerate(self.states):
            name = StateSet.TIERED_STATE_PREFIX+"("+repr(i)+")"
            named_states[name]=state
        return named_states
