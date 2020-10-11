from Code.Config.config import Config

# gnn layer types
PROP_AND_POOL = "prop_and_pool"

# layer options
LAYER_TYPE = "layer_type"
NUM_FEATURES = "num_features"
SAME_WEIGHT_REPEATS = "same_weight_repeats"
DISTINCT_WEIGHT_REPEATS = "distinct_weight_repeats"
LAYER_ARGS = "layer_args"

# module_options
MODULE_TYPE = "module_type"
MODULES = "modules"
PREPARATION_MODULES = "preparation_" + MODULES
MESSAGE_MODULES = "message_" + MODULES
UPDATE_MODULES = "update_" + MODULES

NUM_BASES = "num_bases"
HEADS = "heads"

# layer args
ACTIVATION_TYPE = "activation_type"
ACTIVATION_ARGS = "activation_args"
NEGATIVE_SLOPE = "negative_slope"  # for Leaky_Relu
DROPOUT_RATIO = "dropout_ratio"
NUM_LINEAR_LAYERS = "num_linear_layers"


# prop and pool args
PROPAGATION_TYPE = "prop_type"
POOL_TYPE = "pool_type"
POOL_ARGS = "pool_args"
POOL_RATIO = "ratio"


class GNNConfig(Config):

    def __init__(self):

        super().__init__()
        from torch import nn
        from Code.Models.GNNs.OutputModules.candidate_selection import CandidateSelection
        from Code.Models.GNNs.Layers.CustomLayers.graph_transformer import GraphTransformer

        self.relations_basis_count = 3

        self.global_params = {
            ACTIVATION_TYPE: nn.ReLU,
            DROPOUT_RATIO: 0.5,
        }

        self.use_node_type_embeddings = True

        self.layers = [
            # {
            #     PREPARATION_MODULES: [
            #         # {MODULE_TYPE: RelationalPrep, NUM_BASES: 3}
            #         {MODULE_TYPE: LinearPrep, NUM_LINEAR_LAYERS: 1}
            #     ],
            #     MESSAGE_MODULES:
            #         [{MODULE_TYPE: AttentionModule, HEADS: 8,
            #           ACTIVATION_TYPE: nn.LeakyReLU, ACTIVATION_ARGS: {NEGATIVE_SLOPE: 0.2}}],
            #     # [{MODULE_TYPE: MessageModule}],
            #
            #     UPDATE_MODULES: [{MODULE_TYPE: LinearUpdate, NUM_LINEAR_LAYERS: 1}],
            #     NUM_FEATURES: 400,
            #     SAME_WEIGHT_REPEATS: 1,
            #     DISTINCT_WEIGHT_REPEATS: 1
            # }
            {
                LAYER_TYPE: GraphTransformer,
                HEADS: 8,
                NUM_FEATURES: 400,
                SAME_WEIGHT_REPEATS: 1,
                DISTINCT_WEIGHT_REPEATS: 4
            }
        ]

        self.output_layer = {
            LAYER_TYPE: CandidateSelection,
        }

    def get_gnn_with_constructor_embedder(self, constructor, embedder):
        from Code.Models.GNNs.ContextGNNs.context_gnn import ContextGNN
        from Code.Models.GNNs.ContextGNNs.context_gat import ContextGAT
        from Code.Models.GNNs.ContextGNNs.context_sage import ContextSAGE
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
