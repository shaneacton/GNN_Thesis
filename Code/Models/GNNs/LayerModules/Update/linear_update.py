from Code.Models.GNNs.LayerModules.Update.update_module import UpdateModule
from Code.Models.GNNs.LayerModules.linear_module import LinearModule


class LinearUpdate(UpdateModule, LinearModule):

    def __init__(self, channels, num_linear_layers, activation_type, dropout_ratio, activation_kwargs=None):
        UpdateModule.__init__(self, channels, activation_type, dropout_ratio, activation_kwargs=activation_kwargs)
        self.num_linear_layers = num_linear_layers
        self.projection = self.get_linear_sequence(channels, channels)

    def forward(self, aggr_out):
        return self.projection(aggr_out)