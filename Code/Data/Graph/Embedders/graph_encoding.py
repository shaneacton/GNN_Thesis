from torch_geometric.data import Data

from Code.Data.Graph.context_graph import ContextGraph


class GraphEncoding(Data):

    """
    wrapper around Geometric datapoint which retains its context graph
    """

    def __init__(self, graph: ContextGraph, **kwargs):
        super().__init__(**kwargs)
        self.graph = graph
