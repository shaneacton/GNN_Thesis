from torch_geometric.nn import GATConv

from Code.GNNs.gnn_pool_stack import GNNPoolStack
from Code.HDE.hde_glove import HDEGlove
from Code.Pooling.custom_sag_pool import SAGPool
from Config.config import conf


class HDEPool(HDEGlove):

    def __init__(self, GNN_CLASS=None, POOL_CLASS=None, **kwargs):
        from Code.Training.Utils.model_utils import GNN_MAP
        if GNN_CLASS is None:
            GNN_CLASS = GNN_MAP[conf.gnn_class]
        super().__init__(GNN_CLASS=GNN_CLASS, **kwargs)

    def init_gnn(self, GNN_CLASS):
        if GNN_CLASS == GATConv:
            self.gnn = GNNPoolStack(GNN_CLASS, SAGPool, heads=conf.heads, use_gating=conf.use_gating)
        else:
            self.gnn = GNNPoolStack(GNN_CLASS, SAGPool, use_gating=conf.use_gating)

    def pass_gnn(self, x, example, graph):
        edge_index = graph.edge_index()
        x, _, node_id_map = self.gnn(x, edge_index, graph.candidate_nodes)
        return x, node_id_map


