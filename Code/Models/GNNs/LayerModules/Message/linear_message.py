from Code.Models.GNNs.LayerModules.Message.message_module import MessageModule
from Code.Models.GNNs.LayerModules.linear_module import LinearModule


class LinearMessage(MessageModule, LinearModule):

    def __init__(self, channels, num_linear_layers, activation_type, dropout_ratio):
        MessageModule.__init__(self, channels, activation_type, dropout_ratio)
        self.num_linear_layers = num_linear_layers
        self.projection = self.get_linear_sequence(channels, channels)

    def forward(self, x_j):
        """
        :param x_i: (E, in_channels)
        """
        return self.projection(x_j)
