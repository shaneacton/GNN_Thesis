import inspect
import time

from Code.GNNs.gnn_stack import GNNStack
from Code.HDE.hde_model import HDEModel
from Code.Training.timer import log_time
from Config.config import get_config


class HDERel(HDEModel):

    def __init__(self, BASE_GNN_CLASS=None, **kwargs):
        if BASE_GNN_CLASS is None:
            from Code.Utils.model_utils import GNN_MAP
            BASE_GNN_CLASS = GNN_MAP[get_config().gnn_class]
        super().__init__(GNN_CLASS=BASE_GNN_CLASS, **kwargs)

    def init_gnn(self, BASE_GNN_CLASS):
        init_args = inspect.getfullargspec(BASE_GNN_CLASS.__init__)[0]
        args = {"use_edge_type_embs": True}
        if "heads" in init_args:
            args.update({"heads": get_config().heads})
        self.gnn = GNNStack(BASE_GNN_CLASS, **args)

    def pass_gnn(self, x, graph):
        t = time.time()

        edge_types = graph.edge_types()
        edge_index = graph.edge_index()
        x = self.gnn(x, edge_types=edge_types, edge_index=edge_index)

        log_time("gnn", time.time() - t)
        return x
