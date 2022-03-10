import copy

import torch
from torch import nn

from Config.config import get_config


class Gate(nn.Module):

    def __init__(self, gnn, update_linear, gate_linear):
        super().__init__()
        self.gnn = gnn
        self.update_linear = update_linear
        self.gate_linear = gate_linear
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


class GatedGNN(Gate):

    """
        a wrapper around a gnn instance which makes its node update process gated.
        Copied from: Multi-hop Reading Comprehension across Multiple Documents by
        Reasoning over Heterogeneous Graphs.

        Here the gnn can be any function which takes in node states + other as inputs, and returns new node states
    """

    def __init__(self, gnn):
        size = get_config().hidden_size
        update_linear = nn.Linear(size, size)
        gate_linear = nn.Linear(2*size, size)
        super().__init__(gnn, update_linear, gate_linear)


class SharedGatedGNN(Gate):

    """
        same gate, different GNN + TUF
        if TUF and gate are shared, need special logic
    """

    def __init__(self, gated_gnn: GatedGNN):
        if hasattr(get_config(), "share_tuf_params") and get_config().share_tuf_params \
                and get_config().use_transformer_block:  # todo remove legacy
            tuf_gnn = gated_gnn.gnn  # will be s shared TUF
            from Code.GNNs.gnn_stack import SharedTransGNNLayer
            new_gnn = SharedTransGNNLayer(tuf_gnn)
        else:
            new_gnn = copy.deepcopy(gated_gnn.gnn)

        super().__init__(new_gnn, gated_gnn.update_linear, gated_gnn.gate_linear)
