import torch
from torch.nn import Parameter
from torch_geometric.utils import softmax
import torch.nn.functional as F

from Code.Models.GNNs.LayerModules.message_module import MessageModule


class AttentionModule(MessageModule):

    """
    performs multi-headed attention in the message phase

    heads more similar to the transformer whereby increasing headcount decreases head channel count
        this is opposed to GAT where params scale with headcount
    """

    def __init__(self, in_channels, out_channels, heads=1, negative_slope=0.2, dropout=0):
        super().__init__(in_channels, out_channels)
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.heads = heads
        self.head_channels = out_channels//heads
        if heads * self.head_channels != out_channels:
            raise Exception("channels not divisible by num heads")

        self.att = Parameter(torch.Tensor(1, heads, 2 * self.head_channels))

    def forward(self, edge_index_i, x_i, x_j, size_i):
        # x_j ~ (E, channels)

        x_j = x_j.view(-1, self.heads, self.head_channels)
        # x_j ~ (E, heads, head_channels)

        x_i = x_i.view(-1, self.heads, self.head_channels)
        print("x_j:", x_j.size(), "x_i:", x_i.size())
        # x_i ~ (E, heads, head_channels)
        cat = torch.cat([x_i, x_j], dim=-1)
        alpha = (cat * self.att)
        print("cat:", cat.size(), "alph:", alpha.size())
        alpha = alpha.sum(dim=-1)
        print("alph aft sum:", alpha.size())

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)