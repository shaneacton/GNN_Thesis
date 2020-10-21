from Code.Config.config import Config

from Code.constants import LAYER_TYPE, NUM_FEATURES, SAME_WEIGHT_REPEATS, DISTINCT_WEIGHT_REPEATS, HEADS, \
    ACTIVATION_TYPE, DROPOUT_RATIO


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
