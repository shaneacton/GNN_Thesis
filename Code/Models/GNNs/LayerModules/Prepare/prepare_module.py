from Code.Models.GNNs.LayerModules.layer_module import LayerModule


class PrepareModule(LayerModule):

    """operates on node states in a non topologically aware manner"""

    def __init__(self, in_channels, out_channels, activation_type, dropout_ratio, activation_kwargs=None):
        sizes = [in_channels, out_channels]
        LayerModule.__init__(self, sizes, activation_type, dropout_ratio, activation_kwargs=activation_kwargs)

    def forward(self, x):
        return x
