from typing import Type

from Code.Models.GNNs.ContextGNNs.context_gnn import ContextGNN


class GeometricContextGNN(ContextGNN):

    def add_layer(self, in_features, out_features, layer_type: Type):
        layer = layer_type(in_features, out_features)
        self.layers.append(layer)
        self.layers.append(self.activation)

    def init_layers(self, in_features, layer_type: Type, num_layers=5) -> int:
        for l in range(num_layers):
            self.add_layer(in_features, 300, layer_type)
            in_features = 300
        return in_features

    def pass_layer(self, layer, data):
        if "edge_index" in self.get_method_arg_names(layer.forward):
            x = layer(data.x, data.edge_index)
        else:
            x = layer(data.x)
        data.x = x
        return data