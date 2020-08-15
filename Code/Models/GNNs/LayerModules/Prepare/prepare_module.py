from Code.Models.GNNs.LayerModules.layer_module import LayerModule


class PrepareModule(LayerModule):

    """operates on node states in a non topologically aware manner"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, x):
        return x
