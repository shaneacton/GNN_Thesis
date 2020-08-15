from torch import nn

from Code.Models.GNNs.LayerModules.Prepare.prepare_module import PrepareModule


class LinearPrep(PrepareModule):

    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        self.projection = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return self.projection(x)
