from typing import Optional, Tuple

import torch
from torch import softmax, dropout, Tensor
from torch.nn.functional import linear
from torch.nn.modules.linear import _LinearWithBias
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_, constant_
from torch_geometric.nn import MessagePassing

from torch_geometric.utils import softmax as gat_softmax


class CustomGAT(MessagePassing):

    def __init__(self, in_channels, _, heads, dropout=0.):
        super().__init__()
        self.embed_dim = in_channels
        self.num_heads = heads
        self.dropout = dropout
        self.head_dim = in_channels // heads
        assert self.head_dim * heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = Parameter(torch.empty(3 * in_channels, in_channels))

        self.in_proj_bias = Parameter(torch.empty(3 * in_channels))
        self.out_proj = _LinearWithBias(in_channels, in_channels)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)

        constant_(self.in_proj_bias, 0.)
        constant_(self.out_proj.bias, 0.)

    def forward(self, x, edge_index):
        """x ~ (N, f)"""
        q, k, v = linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        out = self.propagate(edge_index, q=q, k=k, v=v)
        return out

    def message(self, q_i: Tensor, k_j: Tensor, v_j: Tensor, size_i, edge_index_i, edge_index_j) -> Tensor:
        """
            all vectors are shaped ~ (E, f)
            q_i is the outgoing query vectors for each edge.
            k_j is the incoming key vectors for each edge
            v_j is the incoming value vectors for each edge


        """
        e, f = q_i.size()
        head_dim = f // self.num_heads

        q_i = q_i.view(e, self.num_heads, head_dim)
        k_j = k_j.view(e, self.num_heads, head_dim)
        v_j = v_j.view(e, self.num_heads, head_dim)  # (E, h, f/h)

        attn_output_weights = (q_i * k_j)  # (E, h, f/h)
        attn_output_weights = torch.sum(attn_output_weights, dim=-1)  # (E, h)

        """now we need to softmax over the dot product scores. We are maxing wrt each incoming edge for a given node"""
        attn_output_weights = gat_softmax(attn_output_weights, edge_index_i, None, size_i)
        attn_output_weights = dropout(attn_output_weights, p=self.dropout, train=self.training)

        v_j = v_j * attn_output_weights.view(e, self.num_heads, 1)  # (E, h, f/h) * (E, h, 1)  ->  (E, h, f/h)
        assert v_j.size(0) == e and v_j.size(1) == self.num_heads and v_j.size(2) == head_dim
        return v_j.view(e, -1)


    # def forward(self, query, key, value):
    #
    #     return self.multi_head_attention_forward(
    #         query, key, value, self.embed_dim, self.num_heads,
    #         self.in_proj_weight, self.in_proj_bias,
    #         self.dropout, self.out_proj.weight, self.out_proj.bias)
    #
    # def multi_head_attention_forward(self,
    #         query: Tensor,
    #         key: Tensor,
    #         value: Tensor,
    #         embed_dim_to_check: int,
    #         num_heads: int,
    #         in_proj_weight: Tensor,
    #         in_proj_bias: Optional[Tensor],
    #         dropout_p: float,
    #         out_proj_weight: Tensor,
    #         out_proj_bias: Optional[Tensor],
    # ) -> Tensor:
    #     r"""
    #     Args:
    #         query, key, value: map a query and a set of key-value pairs to an output.
    #             See "Attention Is All You Need" for more details.
    #         embed_dim_to_check: total dimension of the model.
    #         num_heads: parallel attention heads.
    #         in_proj_weight, in_proj_bias: input projection weight and bias.
    #         dropout_p: probability of an element to be zeroed.
    #         out_proj_weight, out_proj_bias: the output projection weight and bias.
    #
    #     Shape:
    #         Inputs:
    #         - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
    #           the embedding dimension.
    #         - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
    #           the embedding dimension.
    #         - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
    #           the embedding dimension.
    #         Outputs:
    #         - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
    #           E is the embedding dimension.
    #         - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
    #           L is the target sequence length, S is the source sequence length.
    #     """
    #
    #     tgt_len, bsz, embed_dim = query.size()
    #     assert embed_dim == embed_dim_to_check
    #     assert key.size(0) == value.size(0) and key.size(1) == value.size(1)
    #
    #     head_dim = embed_dim // num_heads
    #     assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    #     scaling = float(head_dim) ** -0.5
    #
    #     if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
    #         # self-attention
    #         q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
    #
    #     elif key is value or torch.equal(key, value):
    #         # encoder-decoder attention
    #         # This is inline in_proj function with in_proj_weight and in_proj_bias
    #         _b = in_proj_bias
    #         _start = 0
    #         _end = embed_dim
    #         _w = in_proj_weight[_start:_end, :]
    #         if _b is not None:
    #             _b = _b[_start:_end]
    #         q = linear(query, _w, _b)
    #
    #         # This is inline in_proj function with in_proj_weight and in_proj_bias
    #         _b = in_proj_bias
    #         _start = embed_dim
    #         _end = None
    #         _w = in_proj_weight[_start:, :]
    #         if _b is not None:
    #             _b = _b[_start:]
    #         k, v = linear(key, _w, _b).chunk(2, dim=-1)
    #
    #     q = q * scaling
    #
    #     q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    #     k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    #     v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    #
    #     src_len = k.size(1)
    #
    #     attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    #     assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]
    #
    #     attn_output_weights = softmax(attn_output_weights, dim=-1)
    #     attn_output_weights = dropout(attn_output_weights, p=dropout_p, train=self.training)
    #
    #     attn_output = torch.bmm(attn_output_weights, v)
    #     assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    #     attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    #     attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    #
    #     return attn_output