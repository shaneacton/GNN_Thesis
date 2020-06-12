import inspect

import torch
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing, RGCNConv
from torch_geometric.utils import remove_self_loops, add_self_loops

from Code.Data.Graph import example_edge_index, example_x
from Code.Training import device


class ExampleSAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(ExampleSAGEConv, self).__init__(aggr='max')  # "Max" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.act = torch.nn.ReLU()
        self.update_lin = torch.nn.Linear(in_channels + out_channels, in_channels, bias=False)
        self.update_act = torch.nn.ReLU()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        print("f1 edge:",edge_index.size())
        edge_index, _ = remove_self_loops(edge_index)
        print("f2 edge:",edge_index.size())
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        print("f3 edge:",edge_index.size())

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index_i):
        # x_j has shape [E, in_channels]
        mask = torch.tensor([1,0,0,1,1,0,0,0,1], dtype=torch.long).to(device)
        mask = mask.view(-1,1)
        prod =  mask * x_j
        print("prod:",prod.size(), prod)
        x_j = self.lin(x_j)
        x_j = self.act(x_j)

        return x_j

    def update(self, aggr_out, x):
        print("update aggr:",aggr_out.size(), "x:", x.size())
        # aggr_out has shape [N, out_channels]

        new_embedding = torch.cat([aggr_out, x], dim=1)

        new_embedding = self.update_lin(new_embedding)
        new_embedding = self.update_act(new_embedding)

        return new_embedding

if __name__ == "__main__":
    conv = ExampleSAGEConv(3, 6).to(device)
    rconv = RGCNConv

    out = conv(example_x, example_edge_index)

    print("out:", out.size())

