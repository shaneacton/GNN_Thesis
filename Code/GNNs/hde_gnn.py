from torch import nn

from Code.GNNs.gated_gnn import GatedGNN
from Code.GNNs.switch_gnn import SwitchGNN
from Config.config import conf


class HDEGNN(nn.Module):

    def __init__(self, in_size, hidden_size, BASE_GNN_CLASS=None, **layer_kwargs):
        if BASE_GNN_CLASS is None:
            from Code.Training.Utils.model_utils import GNN_MAP
            BASE_GNN_CLASS = GNN_MAP[conf.gnn_class]

        super().__init__()

        base = BASE_GNN_CLASS(in_size, hidden_size, **layer_kwargs)
        r_gnn = SwitchGNN(gnn=base)
        self.ggnn = GatedGNN(r_gnn)

    def forward(self, x, graph, **kwargs):
        x = self.ggnn(x, graph=graph, **kwargs)
        return x