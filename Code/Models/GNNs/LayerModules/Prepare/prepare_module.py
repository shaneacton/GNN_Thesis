from Code.Data.Graph.Embedders.graph_encoding import GraphEncoding
from Code.Data.Graph.Embedders.type_embedder import TypeEmbedder
from Code.Models.GNNs.LayerModules.layer_module import LayerModule
from Code.Config import graph_embedding_config
from Code.Training import device


class PrepareModule(LayerModule):

    """operates on node states in a non topologically aware manner"""

    def __init__(self, in_channels, out_channels, activation_type, dropout_ratio, activation_kwargs=None, use_node_type_embeddings=False):

        self.use_node_type_embeddings = use_node_type_embeddings
        sizes = [in_channels, out_channels]
        LayerModule.__init__(self, sizes, activation_type, dropout_ratio, activation_kwargs=activation_kwargs)

        if use_node_type_embeddings:
            self.node_type_embedder = TypeEmbedder(in_channels, graph_feature_type=graph_embedding_config.NODE_TYPES)

    def forward(self, x, encoding):
        if self.use_node_type_embeddings:
            return self.add_node_type_embeddings(x, encoding)

        return x

    def add_node_type_embeddings(self, x, encoding: GraphEncoding):
        type_embeddings = self.node_type_embedder(encoding.types.node_types)
        # print(self, "adding node type embs:", type_embeddings.size())
        x = x + type_embeddings
        return x