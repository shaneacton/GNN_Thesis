from typing import Type

from Code.Data.Graph.graph_encoding import GraphEncoding
from Code.Models.GNNs.ContextGNNs.context_gnn import ContextGNN
from Code.constants import NUM_LAYERS


class GeometricContextGNN(ContextGNN):

    def add_layer(self, in_features, out_features, layer_type: Type):
        layer = layer_type(in_features, out_features)
        self.layers.append(layer)
        if not "activation" in self.__dict__:
            self.init_activation()
        self.layers.append(self.activation)

    def init_layers(self, in_features, layer_type: Type) -> int:
        ContextGNN.init_layers(self, in_features)
        for l in range(self.gnnc.global_params[NUM_LAYERS]):
            self.add_layer(in_features, 400, layer_type)
            in_features = 400
        return in_features

    def pass_layer(self, layer, data: GraphEncoding):
        # print("data.x before:")
        if "edge_index" in self.get_method_arg_names(layer.forward):
            # print("passing", data, "through", layer)
            x = layer(data.x, data.edge_index)
        else:
            x = layer(data.x)
        data.x = x
        return data