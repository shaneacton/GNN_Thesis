import torch
import torch.nn.functional as F
from torch import nn, cat, sigmoid
from torch_geometric.data import Batch
from torch_geometric.nn import TopKPooling, SAGEConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from Code.Data.Graph import example_batch
from Code.Models.GNNs.Abstract.gnn import GNN
from Code.Models.GNNs.Abstract.graph_module import GraphModule
from Code.Models.GNNs.CustomLayers.prop_and_pool_layer import PropAndPoolLayer
from Code.Training import device


class PropAndPool(GNN):
    def __init__(self, in_size):
        super(PropAndPool, self).__init__()

        pnp_args = {"activation_type": nn.ReLU, "prop_type": SAGEConv, "pool_type": TopKPooling,
                    "pool_args": {"ratio": 0.8}}

        self.prop_module = GraphModule([in_size, 128, 128], PropAndPoolLayer, 2, num_hidden_repeats=1,
                                       repeated_layer_args=pnp_args, return_all_outputs=True)

        self.lin1 = nn.Linear(256, 128)
        self.act1 = nn.ReLU()

        self.lin2 = nn.Linear(128, 64)
        self.act2 = nn.ReLU()

        self.lin3 = nn.Linear(64, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, data: Batch):
        kwargs = self.get_required_kwargs_from_batch(data, self.prop_module)
        state_set = self.get_state_set_from_batch(data)

        try:
            _, outs = self.prop_module(**kwargs)
        except Exception as e:
            print("failed to prop through", self.prop_module, "with kwargs:",kwargs)
            raise e
        """sum all the (cat(gmp, gap)) outputs from each module layer"""
        outs = [cat([gmp(x_i, batch), gap(x_i, batch)], dim=1) for x_i, _, _, batch  in outs]
        x = outs[0]
        for i in range(1, len(outs)):
            x += outs[i]

        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = sigmoid(self.lin3(x)).squeeze(1)

        return x


if __name__ == "__main__":
    torch.manual_seed(0)
    pnp = PropAndPool(3).to(device)
    print(pnp)

    out = pnp(example_batch)
    print("pnp_out:",out)
