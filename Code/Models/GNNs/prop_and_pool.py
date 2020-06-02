from typing import Union

import torch.nn.functional as F
from torch import nn, cat, sigmoid
from torch_geometric.data import Batch
from torch_geometric.nn import TopKPooling, SAGEConv, SAGPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from Code.Data.Graph.State.state_set import StateSet
from Code.Models.GNNs.prop_and_pool_layer import PropAndPoolLayer


class PropAndPool(nn.Module):
    def __init__(self, in_size):
        super(PropAndPool, self).__init__()

        self.layer_1 = PropAndPoolLayer([in_size, 128], SAGEConv, TopKPooling, pool_args={"ratio":0.8})
        self.layer_2 = PropAndPoolLayer([128, 128], SAGEConv, TopKPooling, pool_args={"ratio":0.8})
        self.layer_3 = PropAndPoolLayer([128, 128], SAGEConv, TopKPooling, pool_args={"ratio":0.8})

        self.lin1 = nn.Linear(256, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

    def forward(self, data: Batch):
        edge_index, batch = data.edge_index, data.batch
        states = self.get_states(data)

        x= states[StateSet.CURRENT_STATE]
        x = x.squeeze(1)

        x, edge_index, _, batch, = self.layer_1(x, edge_index, None, batch)
        x1 = cat([gmp(x, batch), gap(x, batch)], dim=1)

        x, edge_index, _, batch, = self.layer_2(x, edge_index, None, batch)
        x2 = cat([gmp(x, batch), gap(x, batch)], dim=1)

        x, edge_index, _, batch, = self.layer_3(x, edge_index, None, batch)
        x3 = cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = sigmoid(self.lin3(x)).squeeze(1)

        return x

    def get_states(self, data: Batch):
        return {name: getattr(data, name) for name in StateSet.ALL_STATES if hasattr(data, name)}