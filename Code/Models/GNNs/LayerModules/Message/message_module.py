from Code.Models.GNNs.LayerModules.layer_module import LayerModule


class MessageModule(LayerModule):

    def __init__(self, channels, activation_type, dropout_ratio, activation_kwargs=None):
        LayerModule.__init__(self, [channels] ,activation_type, dropout_ratio, activation_kwargs=activation_kwargs)

    def forward(self, edge_index_i, edge_index_j, x_i, x_j, size_i, encoding, **kwargs):
        """
        :param edge_index_i: (E)
        :param x_i: (E, in_channels)
        :param x_j: (E, in_channels)
        """
        return x_j