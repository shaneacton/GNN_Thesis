import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax

from Code.Models.GNNs.LayerModules.Message.message_module import MessageModule
from Code.Models.GNNs.LayerModules.Message.relational_message import RelationalMessage


class AttentionModule(MessageModule):

    """
    performs multi-headed attention in the message phase

    * Does not apply any linear transformations to the input if no edgewise transformation is applied
    instead a linear transformation should be applied in a preparation module if no edgeise transformations are required

    heads more similar to the transformer whereby increasing headcount decreases head channel count
        this is opposed to GAT where params scale with headcount

    :parameter use_relational_scoring whether or not to switch the scoring functions based on edge type
    :parameter use_edgewise_transformations whether or not to transform edge messages in a
    """

    def __init__(self, channels, heads=1, negative_slope=0.2, dropout=0, num_bases=1,
                 use_relational_scoring=True, use_edgewise_transformations=True):
        super().__init__(channels)
        self.use_edgewise_transformations = use_edgewise_transformations
        self.use_relational_scoring = use_relational_scoring
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.heads = heads
        self.head_channels = channels // heads
        if heads * self.head_channels != channels:
            raise Exception("channels not divisible by num heads")

        if use_relational_scoring:
            # learned function to score connections differently depending on edge type
            self.relational_attention = RelationalMessage(heads, 2 * self.head_channels, num_bases)
        else:
            self.att = Parameter(torch.Tensor(1, heads, 2 * self.head_channels))

        if use_edgewise_transformations:
            self.edgewise_transformations = RelationalMessage(channels, channels, num_bases)

        self.reset_parameters()

    def reset_parameters(self):
        if "att" in self.__dict__:
            glorot(self.att)

    def get_attention_scoring_matrix(self, edge_types):
        if self.use_relational_scoring:
            return self.relational_attention.get_relational_weights(edge_types)  # (E, heads, 2 * head_channels)
        else:
            return self.att  # (1, heads, 2 * head_channels)

    def forward(self, edge_index_i, x_i, x_j, size_i, edge_types):
        # x_j ~ (E, channels)

        if self.use_edgewise_transformations:
            # transforms each message in an edge aware manner
            x_j = self.edgewise_transformations(x_j, edge_types)

        x_j = x_j.view(-1, self.heads, self.head_channels)
        # x_j ~ (E, heads, head_channels)

        x_i = x_i.view(-1, self.heads, self.head_channels)
        # print("x_j:", x_j.size(), "x_i:", x_i.size())
        # x_i ~ (E, heads, head_channels)
        cat = torch.cat([x_i, x_j], dim=-1)  # (E, heads, 2 * head_channels)
        print("att forward x_j=",x_j)
        att = self.get_attention_scoring_matrix(edge_types)
        alpha = (cat * att)  # (E, heads, 2 * head_channels)
        # print("cat:", cat.size(), "alph:", alpha.size())
        alpha = alpha.sum(dim=-1)  # (E, heads)
        # print("alph aft sum:", alpha.size())

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        x_j =  x_j * alpha.view(-1, self.heads, 1)
        return x_j.view(-1, self.heads * self.head_channels)