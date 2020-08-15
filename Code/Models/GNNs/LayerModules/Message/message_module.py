from Code.Models.GNNs.LayerModules.layer_module import LayerModule


class MessageModule(LayerModule):

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

    def forward(self, edge_index_i, x_i, x_j, size_i, **kwargs):
        """

        :param edge_index_i:
        :param x_i:
        :param x_j: has shape [E, in_channels]
        :param size_i:
        :param kwargs:
        :return:
        """
        return x_j
