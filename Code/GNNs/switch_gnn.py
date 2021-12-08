from torch import nn

from Code.HDE.Graph.graph import HDEGraph
from Code.HDE.switch_module import SwitchModule
from Code.constants import GLOBAL
from Config.config import conf


class SwitchGNN(nn.Module):

    """a wrapper around a gnn instance which runs different edges through different gnns and aggregates after"""

    def __init__(self, gnn=None):
        super().__init__()
        self.include_global = conf.use_global_edge_message
        types = ['candidate2candidate', 'candidate2document', 'candidate2entity', 'codocument', 'comention', 'document2entity', 'entity']
        self.gnns = SwitchModule(gnn, types=types, include_global=self.include_global)

    def forward(self, *inputs, graph: HDEGraph = None, **kwargs):
        type_messages = []
        for edge_type in sorted(graph.unique_edge_types):  # sorted to ensure consistent order
            edge_index = graph.edge_index(type=edge_type)
            kwargs.update({"edge_index": edge_index})
            x = self.gnns(*inputs, type=edge_type, **kwargs)
            type_messages.append(x)
        kwargs.update({"edge_index": graph.edge_index()})

        if self.include_global:
            x = self.gnns(*inputs, type=GLOBAL, **kwargs)

            type_messages.append(x)

        x_agg = sum(type_messages) / len(type_messages)

        return x_agg