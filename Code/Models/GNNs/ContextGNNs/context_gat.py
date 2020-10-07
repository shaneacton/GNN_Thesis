from torch_geometric.nn import GATConv

from Code.Models.GNNs.ContextGNNs.context_gnn import ContextGNN


class ContextGAT(ContextGNN):

    def init_layers(self, in_features, num_layers=10) -> int:

        for l in range(num_layers):
            layer = GATConv(in_features, in_features)
            self.layers.append(layer)
        return in_features

    def pass_layer(self, layer, data):
        x = layer(data.x, data.edge_index)
        data.x = x
        return data