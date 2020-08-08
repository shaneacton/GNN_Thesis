import torch
from torch_geometric.data import Data

from Code.Config import GraphEmbeddingConfig
from Code.Data.Graph import TypeMap
from Code.Data.Graph.Types.types import Types
from Code.Data.Graph.context_graph import ContextGraph
from Code.Training import device


class GraphEncoding(Data):

    """
    wrapper around Geometric datapoint which retains its context graph
    """

    def __init__(self, graph: ContextGraph, gec: GraphEmbeddingConfig, types: Types, **kwargs):
        super().__init__(**kwargs)
        self.types: Types = types
        self.gec: GraphEmbeddingConfig = gec  # the config used to embed the given graph
        self.graph: ContextGraph = graph
        self.batch: torch.Tensor = torch.tensor([0] * kwargs['x'].size(0)).to(device)
