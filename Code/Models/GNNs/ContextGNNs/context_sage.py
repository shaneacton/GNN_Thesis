from torch_geometric.nn import SAGEConv

from Code.Models.GNNs.ContextGNNs.geometric_context_gnn import GeometricContextGNN


class ContextSAGE(GeometricContextGNN):

    def init_layers(self, in_features, num_layers=5) -> int:
        return super().init_layers(in_features, SAGEConv, num_layers=num_layers)