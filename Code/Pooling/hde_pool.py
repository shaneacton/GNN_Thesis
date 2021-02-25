from torch_geometric.nn import SAGPooling

from Code.HDE.hde_glove import HDEGlove
from Code.HDE.wikipoint import Wikipoint


class HDEPool(HDEGlove):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pool = SAGPooling(self.hidden_size)

    def forward(self, example: Wikipoint, graph=None):
        x = self.get_graph_features(example)
        graph = self.create_graph(example)
        edge_index = graph.edge_index
        print("x before:", x.size(), "edge:", edge_index.size())

        x, edge_index, _, _, _, _ = self.pool(x, edge_index)
        print("x after:", x.size(), "edge:", edge_index.size())

        return super().forward(example, graph)