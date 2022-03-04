import torch
from torch import dropout, Tensor
from torch.nn import Linear
from torch.nn.functional import linear
from torch.nn.init import xavier_uniform_, constant_
from torch.nn.parameter import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax as gat_softmax, remove_self_loops, add_self_loops

from Config.config import get_config


class CustomGAT(MessagePassing):

    def __init__(self, in_channels, out_channels, heads, dropout=0., add_self_loops: bool = True,
                 output_linear: bool=None, residual_attention: bool=None):

        super().__init__()
        self.embed_dim = in_channels
        self.num_heads = heads
        self.dropout = dropout
        self.head_dim = in_channels // heads
        assert self.head_dim * heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = Parameter(torch.empty(3 * in_channels, in_channels))
        self.in_proj_bias = Parameter(torch.empty(3 * in_channels))

        self.out_proj = Linear(in_channels, out_channels)

        self._reset_parameters()
        self.add_self_loops = add_self_loops
        if residual_attention is None:
            residual_attention = get_config().use_residual_attention
        self.residual_attention = residual_attention
        if residual_attention:
            self.last_attention_scores = None

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)

        constant_(self.in_proj_bias, 0.)
        constant_(self.out_proj.bias, 0.)

    def forward(self, x, edge_index, **kwargs):
        """x ~ (N, f)"""
        # forward_time = time.time()
        q, k, v = linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        scaling = float(self.head_dim) ** -0.5
        q = q * scaling

        if self.add_self_loops:
            num_nodes = x.size(0)
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        out = self.propagate(edge_index, q=q, k=k, v=v, **kwargs)
        out = linear(out, self.out_proj.weight, self.out_proj.bias)
        return out

    def message(self, q_i: Tensor, k_j: Tensor, v_j: Tensor, size_i, index, previous_attention_scores=None) -> Tensor:
        """
            all vectors are shaped ~ (E, f)
            q_i is the outgoing query vectors for each edge.
            k_j is the incoming key vectors for each edge
            v_j is the incoming value vectors for each edge


        """
        # message_time = time.time()
        e, f = q_i.size()
        head_dim = f // self.num_heads

        q_i = q_i.view(e, self.num_heads, head_dim)
        k_j = k_j.view(e, self.num_heads, head_dim)
        v_j = v_j.view(e, self.num_heads, head_dim)  # (E, h, f/h)

        attn_output_weights = (q_i * k_j)  # (E, h, f/h)
        attn_output_weights = torch.sum(attn_output_weights, dim=-1)  # (E, h)

        """now we need to softmax over the dot product scores. We are maxing wrt each incoming edge for a given node"""
        attn_output_weights = gat_softmax(attn_output_weights, index, None, size_i)
        if previous_attention_scores is not None:
            attn_output_weights = attn_output_weights + previous_attention_scores

        # softmax_time = time.time()
        soft_attention_scores = dropout(attn_output_weights, p=self.dropout, train=self.training)

        v_j = v_j * soft_attention_scores.view(e, self.num_heads, 1)  # (E, h, f/h) * (E, h, 1)  ->  (E, h, f/h)
        assert v_j.size(0) == e and v_j.size(1) == self.num_heads and v_j.size(2) == head_dim
        v_j = v_j.view(e, -1)

        # print("message time:", (softmax_time-message_time), "softmax time:", (time.time()-softmax_time))

        if self.residual_attention:
            self.last_attention_scores = attn_output_weights  # temp store to bypass pytorch geometric
            # print("storing residual attentions")
        return v_j