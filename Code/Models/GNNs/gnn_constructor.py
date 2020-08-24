from Code.Config import gnn_config, GNNConfig, GraphConstructionConfig
from Code.Data.Graph.Embedders.graph_embedder import GraphEmbedder


class GNNConstructor:
    """takes in GNN config and outputs a GNN"""

    def __init__(self, gnnc: GNNConfig, gcc: GraphConstructionConfig, graph_embedder: GraphEmbedder):
        self.gnnc = gnnc
        self.gcc = gcc
        self.graph_embedder = graph_embedder

    def get_layers(self, in_features):
        layers = []
        for layer_conf in self.gnnc.layers:
            layer_features = layer_conf[gnn_config.NUM_FEATURES]

            layer = GraphModule([in_features, layer_features, layer_features], layer_conf[gnn_config.LAYER_TYPE],
                                layer_conf[gnn_config.DISTINCT_WEIGHT_REPEATS],
                                num_hidden_repeats=layer_conf[gnn_config.SAME_WEIGHT_REPEATS],
                                repeated_layer_args=layer_conf[gnn_config.LAYER_ARGS])
            layers.append(layer)
            in_features = layer_features

        out_type = self.gnnc.output_layer[gnn_config.LAYER_TYPE]
        layers.append(out_type(in_features))