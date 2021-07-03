import torch
from torch import nn

from Config.config import conf


class GatedGNN(nn.Module):

    """
        a wrapper around a gnn instance which makes its node update process gated.
        Copied from: Multi-hop Reading Comprehension across Multiple Documents by
        Reasoning over Heterogeneous Graphs.

        Here the gnn can be any function which takes in node states + other as inputs, and returns new node states
    """

    def __init__(self, gnn):
        super().__init__()
        self.gnn = gnn
        size = conf.hidden_size
        self.update_linear = nn.Linear(size, size)
        self.gate_linear = nn.Linear(2*size, size)
        self.sig = nn.Sigmoid()

    def forward(self, x, *inputs, **kwargs):
        """gating according to HDE paper"""
        z = self.gnn(x, *inputs, **kwargs)
        u = self.update_linear(x) + z

        g = torch.cat([u, x], dim=-1)
        g = self.sig(self.gate_linear(g))

        u = nn.functional.tanh(u)
        new_x = u * g + x * (1 - g)

        return new_x