from typing import Type

import torch

from Code.Data.Graph.graph_encoding import GraphEncoding
from Code.Models.GNNs.ContextGNNs.context_gnn import ContextGNN


class GeometricContextGNN(ContextGNN):

    def add_layer(self, in_features, out_features, layer_type: Type):
        layer = layer_type(in_features, out_features)
        self.layers.append(layer)
        if not "activation" in self.__dict__:
            self.init_activation()
        self.layers.append(self.activation)

    def init_layers(self, in_features, layer_type: Type, num_layers=5) -> int:
        ContextGNN.init_layers(self, in_features)
        for l in range(num_layers):
            self.add_layer(in_features, 300, layer_type)
            in_features = 300
        return in_features

    def pass_layer(self, layer, data: GraphEncoding):
        # if data.is_batched:
        #     print("passing batched graph", data, "through", layer)
        #     print("e_s:", data.edge_index.size())
        #     print("edge:", data.edge_index)
        #     print("edge min:", torch.min(data.edge_index), "max:", torch.max(data.edge_index))
        if "edge_index" in self.get_method_arg_names(layer.forward):
            x = layer(data.x, data.edge_index)
        else:
            x = layer(data.x)
        data.x = x
        return data