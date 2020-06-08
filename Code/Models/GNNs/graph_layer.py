from typing import List, Type

from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops


class GraphLayer(MessagePassing):
    """wrapper around a propagation layer"""

    def __init__(self, sizes: List[int], layer_type:Type[MessagePassing], activation_type=None, layer_args=None):
        super().__init__()
        self.layer_type = layer_type
        self.sizes = sizes
        self.layer_args = layer_args if layer_args else {}

        self.layer = self.initialise_layer()
        self.activation = activation_type() if activation_type else None

    @property
    def input_size(self):
        return self.sizes[0]

    @property
    def output_size(self):
        return self.sizes[-1]

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def initialise_layer(self):
        if issubclass(self.layer_type, GraphLayer):
            raise Exception("Base Graph layer must contain a GNN primitive layer")

        print("creating layer", self.layer_type,"sizes:",self.sizes, "args:",self.layer_args)
        return self.layer_type(*self.sizes, **self.layer_args)

    def message(self, x_j, **kwargs):
        return self.layer.message(x_j, **kwargs)

    def update(self, inputs, **kwargs):
        return self.layer.update(inputs, **kwargs)

    def forward(self, x, edge_index, edge_types, batch, **kwargs):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        print("x:", x.size(), "batch:",batch,"kwargs:",kwargs)
        # x = self.propagate(edge_index, x=x, edge_types=edge_types, batch=batch, **kwargs)
        x = self.layer(x, edge_index)
        if self.activation:
            return self.activation(x)
        return x
