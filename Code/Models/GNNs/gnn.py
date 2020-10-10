from abc import abstractmethod
from typing import List

from torch import nn

from Code.Config import GNNConfig, ConfigSet, graph_embedding_config
from Code.Config.gnn_config import ACTIVATION_ARGS, ACTIVATION_TYPE, DROPOUT_RATIO
from Code.Data.Graph.Contructors.graph_constructor import GraphConstructor
from Code.Data.Graph.Embedders.graph_embedder import GraphEmbedder
from Code.Data.Graph.Embedders.graph_encoding import GraphEncoding
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

    def init_layers(self, in_features) -> int:  # returns the feature num of the last layer
        if self.configs.gnnc.use_node_type_embeddings:
            self.node_type_embedder = TypeEmbedder(in_features, graph_feature_type=graph_embedding_config.NODE_TYPES)

    def add_node_type_embeddings(self, data: GraphEncoding):
        type_embeddings = self.node_type_embedder(data.types.node_types)
        # print(self, "adding node type embs:", type_embeddings.size())
        data.x = data.x + type_embeddings
        return data

    def _forward(self, data: GraphEncoding) -> GraphEncoding:
        if self.configs.gnnc.use_node_type_embeddings:
            data = self.add_node_type_embeddings(data)
        return data




