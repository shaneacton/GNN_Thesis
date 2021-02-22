from typing import Union, Optional

from torch import nn, Tensor
from torch_geometric.nn import GATConv
from torch_geometric.typing import OptTensor, OptPairTensor, Adj, Size
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_sparse import SparseTensor, set_diag

import torch.nn.functional as F


class RGat(GATConv):

    def __init__(self, in_channels: int, out_channels: int, num_types, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        self.type_embedder = nn.Embedding(num_types, out_channels)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None):
        r"""

        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels

        print("x:", x.size())

        assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
        x_l = x_r = self.lin_l(x).view(-1, H, C)
        print("x_l:", x_l.size())
        alpha_l = (x_l * self.att_l).sum(dim=-1)
        alpha_r = (x_r * self.att_r).sum(dim=-1)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), size=size)

        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        out += self.bias
        return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        print("x_J:", x_j.size())
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)