from torch import nn

from Code.Models.GNNs.LayerModules.Message.message_module import MessageModule


class LinearMessage(MessageModule):

    def __init__(self, channels):
        super().__init__(channels)
        self.lin = nn.Linear(channels, channels)

    def forward(self, edge_index_i, edge_index_j, x_i, x_j, size_i, encoding, **kwargs):
        """
        :param edge_index_i: (E)
        :param x_i: (E, in_channels)
        :param x_j: (E, in_channels)
        """
        return self.lin(x_j)