from Code.Data.Graph.State.channel_state import ChannelState
from Code.Data.Graph.State.state_set import StateSet


class TieredState(ChannelState):
    """
    an array of states where states only communicate up the hierarchy
    eg tier 0 updates as a function of tier 0 only
    but tier 4 updates as a function of  tiers 0-4
    """

    def __init__(self, starting_state, name, num_channels=3):
        super().__init__(starting_state, name, num_channels)

    def get_named_state_tensors(self, sufffix_func=StateSet.TIERED_STATE_NAME):
        return super().get_named_state_tensors(sufffix_func=sufffix_func)
