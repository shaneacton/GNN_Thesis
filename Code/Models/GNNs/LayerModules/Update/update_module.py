from Code.Models.GNNs.LayerModules.layer_module import LayerModule


class UpdateModule(LayerModule):

    def __init__(self, channels, activation_type, dropout_ratio, activation_kwargs=None):
        LayerModule.__init__(self, [channels], activation_type, dropout_ratio, activation_kwargs=activation_kwargs)

    def forward(self, aggr_out, x):
        """
            x is the original node states before message.
            aggr_out is the aggregated message states.
        """
        return aggr_out
