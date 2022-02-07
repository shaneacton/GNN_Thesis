from torch import nn, Tensor
from torch_geometric.nn import MessagePassing


class LinearGNN(MessagePassing):

    def __init__(self, in_size, out_size):
        super().__init__(aggr="mean")
        self.message_transform = nn.Linear(in_size, out_size)

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        return out

    def message(self, x_j: Tensor) -> Tensor:
        return self.message_transform(x_j)