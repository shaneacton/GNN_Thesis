# gnn layer types
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


class GNNConfig:

    def __init__(self):

        from torch import nn
        from torch_geometric.nn import TopKPooling, SAGEConv
        from Code.Models.GNNs.CustomLayers.prop_and_pool_layer import PropAndPoolLayer

        self.layers = [
            {
                LAYER_TYPE: PropAndPoolLayer,
                NUM_FEATURES: 400,
                SAME_WEIGHT_REPEATS: 1,
                DISTINCT_WEIGHT_REPEATS: 1,
                LAYER_ARGS : {ACTIVATION_TYPE: nn.ReLU, PROPAGATION_TYPE: SAGEConv, POOL_TYPE: TopKPooling,
                POOL_ARGS: {POOL_RATIO: 0.8}}
            },
            {

            }
        ]




