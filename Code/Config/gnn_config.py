# gnn layer types
from torch_geometric.nn import GATConv

from Code.Config.config import Config
from Code.Models.GNNs.LayerModules.Prepare.linear_prep import LinearPrep
from Code.Models.GNNs.LayerModules.Prepare.prepare_module import PrepareModule

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
        from torch_geometric.nn import RGCNConv
        from Code.Models.GNNs.LayerModules.Message.attention_module import AttentionModule
        from Code.Models.GNNs.LayerModules.Message.relational_message import RelationalMessage
        from Code.Models.GNNs.LayerModules.Prepare.relational_prep import RelationalPrep
        from Code.Models.GNNs.LayerModules.update_module import UpdateModule
        from Code.Models.GNNs.OutputModules.candidate_selection import CandidateSelection

        self.relations_basis_count = 3

        self.layers = [
            # {
            #     LAYER_TYPE: PropAndPoolLayer,
            #     NUM_FEATURES: 400,
            #     SAME_WEIGHT_REPEATS: 1,
            #     DISTINCT_WEIGHT_REPEATS: 1,
            #     LAYER_ARGS : {ACTIVATION_TYPE: nn.ReLU, PROPAGATION_TYPE: SAGEConv, POOL_TYPE: TopKPooling,
            #     POOL_ARGS: {POOL_RATIO: 0.8}}
            # }
            # {
            #     LAYER_TYPE: GATConv,
            #     NUM_FEATURES: 400,
            #     SAME_WEIGHT_REPEATS: 1,
            #     DISTINCT_WEIGHT_REPEATS: 1,
            #     LAYER_ARGS: {ACTIVATION_TYPE: nn.ReLU}
            # }
            # ,

            {
                PREPARATION_MODULES: [
                    # {MODULE_TYPE: RelationalPrep, NUM_BASES: 3}
                    {MODULE_TYPE: LinearPrep}
                ],
                MESSAGE_MODULES: [
                    {MODULE_TYPE: AttentionModule, HEADS: 8}],
                UPDATE_MODULES: [{MODULE_TYPE: UpdateModule}],

                NUM_FEATURES: 400,
                SAME_WEIGHT_REPEATS: 12,
                DISTINCT_WEIGHT_REPEATS: 1,
            }
        ]

        self.output_layer = {
            LAYER_TYPE: CandidateSelection,
        }


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
