from typing import List

from torch import nn

import Code.constants
from Code.Config import GNNConfig, ConfigSet
from Code.constants import ACTIVATION_TYPE, ACTIVATION_ARGS, DROPOUT_RATIO
from Code.Data.Graph.Embedders.graph_encoding import GraphEncoding
# from Code.Data.Graph.Embedders.position_embedder import PositionEmbedder
from Code.Data.Graph.Embedders.type_embedder import TypeEmbedder
from Code.Models.GNNs.gnn_component import GNNComponent


class GNN(GNNComponent, nn.Module):

    def __init__(self, sizes: List[int], gnnc: GNNConfig, configs: ConfigSet = None):
        nn.Module.__init__(self)

        act_args = gnnc.global_params[ACTIVATION_ARGS] if ACTIVATION_ARGS in gnnc.global_params else None
        act_type = gnnc.global_params[ACTIVATION_TYPE]
        dropout = gnnc.global_params[DROPOUT_RATIO]

        GNNComponent.__init__(self, sizes, act_type, dropout, activation_kwargs=act_args)

        self.configs: ConfigSet = configs
        if not self.configs:
            self.configs = ConfigSet(config=gnnc)
        elif not self.configs.gnnc:
            self.configs.add_config(gnnc)

        if self.configs.gnnc.use_node_type_embeddings:
            self.node_type_embedder = None

        # if configs.gec.use_absolute_positional_embeddings:
        #     self.positional_embedder = None

    def init_layers(self, in_features) -> int:  # returns the feature num of the last layer
        if self.configs.gnnc.use_node_type_embeddings:
            self.node_type_embedder = TypeEmbedder(in_features, graph_feature_type=Code.constants.NODE_TYPES)

        # if self.configs.gec.use_absolute_positional_embeddings:
        #     self.positional_embedder = PositionEmbedder(in_features, self.configs.gcc, self.configs.gec)

    def add_node_type_embeddings(self, data: GraphEncoding):
        type_embeddings = self.node_type_embedder(data.types.node_types)
        # print(self, "adding node type embs:", type_embeddings.size())
        data.x = data.x + type_embeddings
        return data

    # def add_positional_embeddings(self, data: GraphEncoding):
    #     # print("positions:", data.node_positions)
    #     position_embeddings = self.positional_embedder(data.node_positions)
    #     # print(self, "adding positional embs:", position_embeddings.size())
    #     data.x = data.x + position_embeddings
    #     return data

    def _forward(self, data: GraphEncoding) -> GraphEncoding:
        if self.configs.gnnc.use_node_type_embeddings:
            data = self.add_node_type_embeddings(data)

        # if self.configs.gec.use_absolute_positional_embeddings:
        #     data = self.add_positional_embeddings(data)
        return data




