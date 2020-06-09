import inspect

import torch
import torch.nn.functional as F
from torch import nn, cat, sigmoid
from torch_geometric.data import Batch
from torch_geometric.nn import TopKPooling, SAGEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from Code.Models.GNNs.graph_layer import GraphLayer
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
        kwargs = self.get_required_kwargs_from_batch(data, self.layer_1)

        x, edge_index, _, batch, = self.layer_1(**kwargs)
        x1 = cat([gmp(x, batch), gap(x, batch)], dim=1)

        kwargs.update({"x":x, "edge_index":edge_index, "batch": batch})
        x, edge_index, _, batch, = self.layer_2(**kwargs)
        x2 = cat([gmp(x, batch), gap(x, batch)], dim=1)

        kwargs.update({"x":x, "edge_index":edge_index, "batch": batch})
        x, edge_index, _, batch, = self.layer_3(**kwargs)
        x3 = cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = sigmoid(self.lin3(x)).squeeze(1)

        return x

    def get_required_kwargs_from_batch(self, data: Batch, layer: GraphLayer):
        """
        plucks params out of the data object which are needed by the given layer
        """
        layer_args = layer.get_base_layer_all_arg_names()
        data_args = data.__dict__
        present_args = {arg: data_args[arg] for arg in layer_args if arg in data_args.keys()}
        return present_args

if __name__ == "__main__":
    from torch_geometric.data import Data

    x = torch.tensor([[2, 1, 3], [5, 6, 4], [3, 7, 5], [12, 0, 6]], dtype=torch.float)
    y = torch.tensor([0, 1, 0, 1], dtype=torch.float)

    edge_index = torch.tensor([[0, 2, 1, 0, 3],
                               [3, 1, 0, 1, 2]], dtype=torch.long)

    edge_types = torch.tensor([[0, 2, 1, 0, 3]], dtype=torch.long)

    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_types)
    # data = Data(x=x, y=y, edge_index=edge_index)

    batch = Batch.from_data_list([data, data])

    pnp = PropAndPool(3)

    out = pnp(batch)
    print("pnp_out:",out.size())
