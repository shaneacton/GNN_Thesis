from Code.Config import gnn_config
from Code.Data.Graph.Embedders.graph_encoding import GraphEncoding
from Code.Models.GNNs.ContextGNNs.context_gnn import ContextGNN
from Code.Models.GNNs.graph_module import GraphModule


class ConfigurableContextGNN(ContextGNN):

    def pass_layer(self, layer, data: GraphEncoding):
        return layer(data)

    def init_layers(self, in_features):
        """
        creates layers based on the gnn config provided as well as the sampled in features size.
        returns the number of features in the last layer for the output layer
        """
        for layer_conf in self.gnnc.layers:
            layer_features = layer_conf[gnn_config.NUM_FEATURES]

            layer = GraphModule([in_features, layer_features, layer_features], layer_conf, self.gnnc)
            self.layers.append(layer)
            in_features = layer_features

        return in_features