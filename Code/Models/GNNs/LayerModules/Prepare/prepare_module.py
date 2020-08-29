from Code.Models.GNNs.LayerModules.layer_module import LayerModule


class PrepareModule(LayerModule):

    """operates on node states in a non topologically aware manner"""

    def __init__(self, in_channels, out_channels, activation_type, dropout_ratio, activation_kwargs=None):
        LayerModule.__init__(self, activation_type, dropout_ratio, activation_kwargs=activation_kwargs)
        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, x):
        return x
