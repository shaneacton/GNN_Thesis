from Code.Models.GNNs.LayerModules.Prepare.prepare_module import PrepareModule
from Code.Models.GNNs.LayerModules.linear_module import LinearModule


class LinearPrep(PrepareModule, LinearModule):

    def __init__(self, in_channels, out_channels, num_linear_layers, activation_type, dropout_ratio,
                 activation_kwargs=None, use_node_type_embeddings=True):

        PrepareModule.__init__(self, in_channels, out_channels, activation_type, dropout_ratio,
                               activation_kwargs=activation_kwargs, use_node_type_embeddings=use_node_type_embeddings)
        self.num_linear_layers = num_linear_layers

        self.projection = self.get_linear_sequence(in_channels, out_channels)

    def forward(self, x, encoding):
        if self.use_node_type_embeddings:
            x = self.add_node_type_embeddings(x, encoding)
        return self.activate(self.projection(x))
