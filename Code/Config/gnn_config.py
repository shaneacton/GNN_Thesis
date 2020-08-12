# gnn layer types
from torch_geometric.nn import GATConv

from Code.Config.config import Config

PROP_AND_POOL = "prop_and_pool"

# layer options
LAYER_TYPE = "layer_type"
NUM_FEATURES = "num_features"
SAME_WEIGHT_REPEATS = "same_weight_repeats"
DISTINCT_WEIGHT_REPEATS = "distinct_weight_repeats"
LAYER_ARGS = "layer_args"

# layer args
ACTIVATION_TYPE = "activation_type"

# prop and pool args
PROPAGATION_TYPE = "prop_type"
POOL_TYPE = "pool_type"
POOL_ARGS = "pool_args"
POOL_RATIO = "ratio"


class GNNConfig(Config):

    def __init__(self):

        super().__init__()
        from torch import nn
        from torch_geometric.nn import SAGEConv

        self.layers = [
            # {
            #     LAYER_TYPE: PropAndPoolLayer,
            #     NUM_FEATURES: 400,
            #     SAME_WEIGHT_REPEATS: 1,
            #     DISTINCT_WEIGHT_REPEATS: 1,
            #     LAYER_ARGS : {ACTIVATION_TYPE: nn.ReLU, PROPAGATION_TYPE: SAGEConv, POOL_TYPE: TopKPooling,
            #     POOL_ARGS: {POOL_RATIO: 0.8}}
            # }
            {
                LAYER_TYPE: GATConv,
                NUM_FEATURES: 400,
                SAME_WEIGHT_REPEATS: 1,
                DISTINCT_WEIGHT_REPEATS: 1,
                LAYER_ARGS: {ACTIVATION_TYPE: nn.ReLU}
            }
        ]

    def get_gnn_with_constructor_embedder(self, constructor, embedder):
        from Code.Models.GNNs.context_gnn import ContextGNN
        from Code.Training import device
        gnn = ContextGNN(constructor, embedder, self).to(device=device)
        return gnn

    def get_gnn(self, gcc, gec):
        from Code.Training import device
        constructor = gcc.get_graph_constructor()
        embedder = gec.get_graph_embedder(gcc).to(device)
        return self.get_gnn_with_constructor_embedder(constructor, embedder)
