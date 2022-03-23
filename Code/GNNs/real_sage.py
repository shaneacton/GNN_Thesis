import torch
from torch import nn, Tensor
from torch_geometric.nn import MessagePassing


class RealSAGE(MessagePassing):

    def __init__(self, in_size, out_size):
        super().__init__(aggr="mean")
        self.concat_transform = nn.Linear(in_size * 2, out_size)

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def update(self, agg_msg: Tensor, x: Tensor) -> Tensor:
        # print("linsage update. agg:", agg_msg.size(), "x:", x.size())
        concat = torch.cat([x, agg_msg], dim=-1)
        return agg_msg + self.concat_transform(concat)