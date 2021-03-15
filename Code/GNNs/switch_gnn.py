from torch import nn

from Code.HDE.Graph.graph import HDEGraph
from Code.HDE.switch_module import SwitchModule, GLOBAL
from Config.config import conf


class SwitchGNN(nn.Module):

    """a wrapper around a gnn instance which runs different edges through different gnns and aggregates after"""

    def __init__(self, in_size=None, hidden_size=None, BASE_GNN_CLASS=None, gnn=None, **layer_kwargs):
        super().__init__()
        self.include_global = conf.use_global_edge_message
        self.hidden_size = hidden_size
        types = ['candidate2candidate', 'candidate2document', 'candidate2entity', 'codocument', 'comention', 'document2entity', 'entity']
        if gnn is None:
            gnn = BASE_GNN_CLASS(in_size, hidden_size, **layer_kwargs)

        self.gnns = SwitchModule(gnn, types=types, include_global=self.include_global)

    def forward(self, *inputs, graph: HDEGraph = None, **kwargs):
        type_messages = []
        for edge_type in graph.unique_edge_types:
            edge_index = graph.edge_index(type=edge_type)
            x = self.gnns(*inputs, type=edge_type, edge_index=edge_index, **kwargs)

            type_messages.append(x)
        if self.include_global:
            edge_index = graph.edge_index()
            x = self.gnns(*inputs, type=GLOBAL, edge_index=edge_index, **kwargs)

            type_messages.append(x)

        x_agg = sum(type_messages) / len(type_messages)
        return x_agg

