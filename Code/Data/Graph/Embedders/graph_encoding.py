from typing import List, Union

import torch
from torch_geometric.data import Batch

from Code.Data.Graph.Types.types import Types
from Code.Data.Graph.context_graph import QAGraph
from Code.Training import device


class GraphEncoding(Batch):

    """
    initialise as a single data point. use Batch.from_data_list to group
    wrapper around Geometric datapoint batch which retains its context graph
    """

    def __init__(self, graph: QAGraph, gec, types: Types, batch=None, generate_batch=False, **kwargs):
        super().__init__(**kwargs)
        self.types: Union[Types, List[Types]] = types
        self.gec = gec  # the config used to embed the given graph
        self.graph: Union[QAGraph, List[QAGraph]] = graph
        if batch is not None and generate_batch:
            raise Exception()
        if batch is not None:
            self.batch=batch
        if generate_batch:
            self.batch: torch.Tensor = torch.tensor([0] * kwargs['x'].size(0)).to(device)

        # self.set_positional_window_sizes()
        self.layer = 0

    @staticmethod
    def from_geometric_batch(geo_batch: Batch):
        geo_args = geo_batch.__dict__
        geo_args["gec"] = geo_args["gec"][0]
        geo_args["layer"] = geo_args["layer"][0]

        batch_encoding = GraphEncoding(**geo_args)
        return batch_encoding

    # @property
    # def node_positions(self):
    #     if isinstance(self.graph, QAGraph):
    #         return self.graph.node_positions
    #     if isinstance(self.graph, List):
    #         positions = []
    #         for graph in self.graph:
    #             positions.extend(graph.node_positions)
    #         return positions
    #     raise Exception()

    @property
    def node_types(self):
        if isinstance(self.types, Types):
            return self.types.node_types
        if isinstance(self.types, List):
            types = [types.node_types for types in self.types]
            return torch.cat(types)

    @property
    def edge_types(self):
        return self.types.edge_types

    # def set_positional_window_sizes(self):
    #     """
    #         these values cannot be set at graph construction time since
    #         the window size should be independent of graph construction
    #     """
    #     def set_window_size(graph):
    #         for pos in graph.node_positions:
    #             if not pos:
    #                 continue
    #             pos.window_size = self.gec.relative_embeddings_window_per_level[pos.sequence_level]
    #
    #     if isinstance(self.graph, List):
    #         for g in self.graph:
    #            set_window_size(g)
    #     if isinstance(self.graph, QAGraph):
    #         set_window_size(self.graph)