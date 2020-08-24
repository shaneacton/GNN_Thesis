import torch
from torch_geometric.data import Data

from Code.Data.Graph.Types.types import Types
from Code.Data.Graph.context_graph import ContextGraph
from Code.Training import device


class GraphEncoding(Data):

    """
    wrapper around Geometric datapoint which retains its context graph
    """

    def __init__(self, graph: ContextGraph, gec, types: Types, **kwargs):
        super().__init__(**kwargs)
        self.types: Types = types
        self.gec = gec  # the config used to embed the given graph
        self.graph: ContextGraph = graph
        self.batch: torch.Tensor = torch.tensor([0] * kwargs['x'].size(0)).to(device)
        self.set_positional_window_sizes()

    @property
    def node_types(self):
        return self.types.node_types

    @property
    def edge_types(self):
        return self.types.edge_types

    def set_positional_window_sizes(self):
        """these values cannot be set at graph construction time since the window size should be independent"""
        for pos in self.graph.node_positions:
            pos.window_size = self.gec.relative_embeddings_window_per_level[pos.sequence_level]
