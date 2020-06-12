from typing import Type, List

from torch_geometric.nn import MessagePassing

from Code.Data.Graph.State.state_set import StateSet
from Code.Models.GNNs.abstract.graph_layer import GraphLayer


class StateMultiplexer(GraphLayer):

    """
    takes in multiple states as input, concatenates each input and feeds through single graph layer
    Layer shapes itself during the first forward call based on the states it
    """

    def __init__(self, sizes: List[int], state_names:List[str], layer_type: Type[MessagePassing], layer_args=None):
        if len(sizes) != 1:
            raise Exception("sizes should contain [out_size] only - in_size is determined dynamically, given sizes: " + repr(sizes))
        super().__init__(None, layer_type, init_layer=False, layer_args=layer_args)
        self.state_names = state_names
        self.out_size = sizes[0]

        self.layer_initialised = False

    def _initialise_layer(self, stateset: StateSet):
        states = stateset.get

    def forward(self, x, edge_index, batch, **kwargs):
        stateset: StateSet = kwargs[StateSet.STATE_SET]
        if not self.layer_initialised:
            self._initialise_layer(stateset)

        return super(StateMultiplexer, self).forward(x, edge_index, batch, **kwargs)