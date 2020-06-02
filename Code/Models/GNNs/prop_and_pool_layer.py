from typing import Union, List

from torch import nn
from torch_geometric.nn import SAGPooling, TopKPooling, SAGEConv

from Code.Models.GNNs.graph_layer import GraphLayer


class PropAndPoolLayer(GraphLayer):
    """
    does one step of propagation then one pool
    """
    def __init__(self, sizes: List[int], prop_type, pool_type: Union[TopKPooling, SAGPooling],
                 activation_type=None, prop_args=None, pool_args=None):
        super().__init__(sizes, PropAndPoolLayer)
        prop_args = prop_args if prop_args else {}
        pool_args = pool_args if pool_args else {}
        self.prop = GraphLayer(sizes, prop_type, activation_type=activation_type, layer_args=prop_args)
        self.pool = pool_type(sizes[1], **pool_args)

    def initialise_layer(self):
        return None

    def forward(self, state, edge_index, edge_attributes, batch):
        state = self.prop(state, edge_index)  # cannot wrap these two in a seq
        state, edge_index, edge_attributes, batch, _, _ = self.pool(state, edge_index, edge_attributes, batch)
        return state, edge_index, edge_attributes, batch


if __name__ == "__main__":
    pnp = PropAndPoolLayer([100,200], SAGEConv, TopKPooling, activation_type=nn.ReLU, pool_args={"ratio":0.8})
    print(pnp)