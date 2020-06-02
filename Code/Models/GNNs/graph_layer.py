from typing import List

from torch import nn


class GraphLayer(nn.Module):
    """wrapper around a propagation layer"""

    def __init__(self, sizes: List[int], layer_type:type, activation_type=None, layer_args=None):
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
        return self.layer_type(*self.sizes, **self.layer_args)

    def forward(self, state, edge_index, edge_attributes, batch):
        state = self.layer(state, edge_index)
        if self.activation:
            return self.activation(state)
        return state