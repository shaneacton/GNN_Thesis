from torch import Tensor
from torch_geometric.nn import MessagePassing


class SAGECore(MessagePassing):

    """basically the identity core. Does nothing except message passing"""

    def __init__(self, in_size, out_size):
        super().__init__(aggr="mean")

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, x=x)
        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def update(self, agg_msg: Tensor, x: Tensor) -> Tensor:
        return agg_msg