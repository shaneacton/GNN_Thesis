import torch
from torch_geometric.nn import GATConv

from Code.Models.GNNs.ContextGNNs.context_gnn import ContextGNN
from Code.Models.GNNs.gnn_component import GNNComponent


class ContextGAT(ContextGNN):

    def init_layers(self, in_features, num_layers=5) -> int:

        for l in range(num_layers):
            layer = GATConv(in_features, in_features)
            self.layers.append(layer)
            self.layers.append(torch.nn.ReLU())
        return in_features

    def pass_layer(self, layer, data):
        if "edge_index" in GNNComponent.get_method_arg_names(layer.forward):
            x = layer(data.x, data.edge_index)
        else:
            x = layer(data.x)
        data.x = x
        return data