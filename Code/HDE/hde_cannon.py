from torch_geometric.nn import GATConv

from Code.GNNs.gnn_stack import GNNStack
from Code.GNNs.hde_gnn import HDEGNN
from Code.HDE.hde_glove import HDEGlove
from Config.config import conf


class HDECannon(HDEGlove):

    def __init__(self, BASE_GNN_CLASS=None, **kwargs):
        if BASE_GNN_CLASS is None:
            from Code.Utils.model_utils import GNN_MAP
            BASE_GNN_CLASS = GNN_MAP[conf.gnn_class]
        super().__init__(GNN_CLASS=BASE_GNN_CLASS, **kwargs)

    def init_gnn(self, BASE_GNN_CLASS):
        if BASE_GNN_CLASS == GATConv:
            self.gnn = GNNStack(HDEGNN, heads=conf.heads, BASE_GNN_CLASS=BASE_GNN_CLASS)
        else:
            self.gnn = GNNStack(HDEGNN, BASE_GNN_CLASS=BASE_GNN_CLASS)

    def pass_gnn(self, x, example, graph):
        return self.gnn(x, graph=graph)