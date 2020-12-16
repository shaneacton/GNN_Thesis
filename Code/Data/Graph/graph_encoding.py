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

    def __init__(self, graphs: QAGraph, gec, types: Types, batch=None, generate_batch=False, **kwargs):
        super().__init__(**kwargs)
        self.types: Types = types
        self.gec = gec  # the config used to embed the given graph
        self.graphs: Union[QAGraph, List[QAGraph]] = graphs
        if batch is not None and generate_batch:
            raise Exception()
        if batch is not None:
            self.batch=batch
        if generate_batch:
            self.batch: torch.Tensor = torch.tensor([0] * kwargs['x'].size(0)).to(device)

        # self.set_positional_window_sizes()
        self.layer = 0

    @staticmethod
    def batch(encodings):
        encodings: List[GraphEncoding] = encodings
        batch = Batch.from_data_list(encodings)
        encoding = GraphEncoding.from_geometric_batch(batch)
        return encoding

    @staticmethod
    def from_geometric_batch(geo_batch: Batch):
        """
            we require a few extra processes on top of the pytorch geometrics batching system
            graphs is the only att which should remain a list, the types should be combined to one
        """
        geo_args = geo_batch.__dict__
        geo_args["layer"] = geo_args["layer"][0]
        geo_args["gec"] = geo_args["gec"][0]
        geo_args["types"] = Types.from_types_list(geo_args["types"])

        batch_encoding = GraphEncoding(**geo_args)
        return batch_encoding

    @property
    def is_batched(self):
        return isinstance(self.graphs, List) and len(self.graphs) > 1

    @property
    def graph(self):
        if not self.is_batched:
            raise Exception("cannot call graph on batched encoding")
        return self.graphs

    @property
    def sample_graph(self):
        if isinstance(self.graphs, List):
            return self.graphs[0]
        return self.graphs

    @property
    def node_types(self):
        if isinstance(self.types, Types):
            return self.types.node_types
        raise Exception()

    @property
    def edge_types(self):
        if isinstance(self.types, Types):
            return self.types.edge_types
        raise Exception()