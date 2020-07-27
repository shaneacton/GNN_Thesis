import torch
from torch_geometric.data import Data

from Code.Config import GraphEmbeddingConfig
from Code.Data.Graph.context_graph import ContextGraph
from Code.Training import device


class GraphEncoding(Data):

    """
    wrapper around Geometric datapoint which retains its context graph
    """

    def __init__(self, graph: ContextGraph, gec: GraphEmbeddingConfig, **kwargs):
        super().__init__(**kwargs)
        self.gec = gec  # the config used to embed the given graph
        self.graph = graph
        self.batch = torch.tensor([0] * kwargs['x'].size(0)).to(device)
