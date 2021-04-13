from torch_geometric.nn import GATConv

from Code.GNNs.gnn_stack import GNNStack
from Code.HDE.hde_glove import HDEGlove
from Config.config import conf


class HDERel2(HDEGlove):

    def __init__(self, BASE_GNN_CLASS=None, **kwargs):
        if BASE_GNN_CLASS is None:
            from Code.Utils.model_utils import GNN_MAP
            BASE_GNN_CLASS = GNN_MAP[conf.gnn_class]
        super().__init__(GNN_CLASS=BASE_GNN_CLASS, **kwargs)

    def init_gnn(self, BASE_GNN_CLASS):
        if BASE_GNN_CLASS == GATConv:
            self.gnn = GNNStack(BASE_GNN_CLASS, heads=conf.heads, use_edge_type_embs=True)
        else:
            self.gnn = GNNStack(BASE_GNN_CLASS, use_edge_type_embs=True)

    def pass_gnn(self, x, example, graph):
        edge_types = graph.edge_types()
        edge_index = graph.edge_index()
        return self.gnn(x, edge_types=edge_types, edge_index=edge_index)