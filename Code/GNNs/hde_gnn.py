from torch import nn
from torch_geometric.nn import SAGEConv

from Code.GNNs.gated_gnn import GatedGNN
from Code.GNNs.switch_gnn import SwitchGNN


class HDEGNN(nn.Module):

    def __init__(self, in_size, hidden_size, BASE_GNN_CLASS=SAGEConv):
        super().__init__()

        base = BASE_GNN_CLASS(in_size, hidden_size)
        r_gnn = SwitchGNN(base)
        self.ggnn = GatedGNN(r_gnn)

    def forward(self, x, graph, **kwargs):
        x = self.ggnn(x, graph=graph, **kwargs)
        return x