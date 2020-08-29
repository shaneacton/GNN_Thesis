from Code.Models.GNNs.LayerModules.layer_module import LayerModule


class AggregateModule(LayerModule):

    def __init__(self, activation_type, dropout_ratio):
        LayerModule.__init__(self, activation_type, dropout_ratio)
