from typing import Tuple, Union

import torch
from torch import Tensor, nn
from torch_geometric.nn import GATConv


class Gat(GATConv):

    def __init__(self, in_channels: Union[int, Tuple[int, int]], out_channels: int, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        self.lin = nn.Linear(2 * out_channels, out_channels)
        self.act = nn.ReLU()

    def update(self, inputs: Tensor, x: Tensor) -> Tensor:
        """
            combines the agg messages with the original states to give assymetry and allow for preservation of
            node specific info such as positions
        """
        x_l, _ = x
        # print("custom gat got aggout:", inputs.size() , "x_l:", x_l.size())
        cat = torch.cat([inputs, x_l], dim=2)
        out = self.act(self.lin(cat))
        return out