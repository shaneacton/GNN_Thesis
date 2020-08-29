from torch import nn

from Code.Models.GNNs.LayerModules.layer_module import LayerModule


class LinearModule(LayerModule):

    def __init__(self, num_linear_layers, activation_type, dropout_ratio):
        LayerModule.__init__(self, activation_type, dropout_ratio)
        self.num_linear_layers = num_linear_layers

    def get_linear_sequence(self, in_channels, out_channels):
        """returns a sequence of linear layers with activations inbetween, not at the end"""
        layers = []
        for i in range(self.num_linear_layers - 1):
            layers.append(nn.Linear(in_channels, out_channels))
            layers.append(self.activation)
            in_channels = out_channels
        layers.append(nn.Linear(in_channels, out_channels))  # no act after last lin, as in Transformer
        return nn.Sequential(*layers)
