from typing import Dict

from torch import Tensor

from Code.Data.Graph.State.state import State
from Code.Data.Graph.State.state_set import StateSet


class ChannelState(State):
    """
    simply an array of states which are converted into separate named states
    Each state represents a separate channel
    """

    def __init__(self, starting_state, name, num_channels=3):
        super().__init__(name)
        self.states= [starting_state] * num_channels

    def get_named_state_tensors(self, sufffix_func=StateSet.CHANNEL_STATE_NAME):
        named_states: Dict[str: Tensor] = {}
        for i, state in enumerate(self.states):
            name = self.name + "_" + sufffix_func(i)
            named_states[name]=state
        return named_states
