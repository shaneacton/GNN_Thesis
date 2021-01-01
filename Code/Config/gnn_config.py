from Code.Config.config import Config
from Code.Models.GNNs.Custom.asymmetrical_gat import AsymGat

from Code.constants import LAYER_TYPE, NUM_FEATURES, SAME_WEIGHT_REPEATS, DISTINCT_WEIGHT_REPEATS, HEADS, \
    ACTIVATION_TYPE, DROPOUT_RATIO, NUM_LAYERS


class GNNConfig(Config):

    def __init__(self):

        super().__init__()
        from torch import nn

        self.global_params = {
            ACTIVATION_TYPE: nn.ReLU,
            DROPOUT_RATIO: 0,
            NUM_LAYERS: 2
        }
        self.layer_type = AsymGat  # GATConv
        #
        self.use_node_type_embeddings = False

    def get_gnn_with_constructor_embedder(self, constructor, embedder):
        from Code.Models.GNNs.ContextGNNs.context_gat import ContextGAT
        from Code.Config import ConfigSet

        from Code.Training import device
        configs = ConfigSet([constructor.gcc, embedder.gec])
        gnn = ContextGAT(constructor, embedder, self, configs=configs).to(device=device)
        return gnn

    def get_gnn(self, gcc, gec):
        from Code.Training import device
        constructor = gcc.get_graph_constructor()
        embedder = gec.get_graph_embedder(gcc).to(device)
        return self.get_gnn_with_constructor_embedder(constructor, embedder)
