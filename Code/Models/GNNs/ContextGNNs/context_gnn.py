from abc import ABC, abstractmethod
from typing import Union

from torch import nn
from torch_geometric.data import Batch

from Code.Config import GNNConfig
from Code.Config.config_set import ConfigSet
from Code.Config.gnn_config import ACTIVATION_TYPE, DROPOUT_RATIO, ACTIVATION_ARGS
from Code.Data.Graph.Contructors.graph_constructor import GraphConstructor
from Code.Data.Graph.Embedders.graph_embedder import GraphEmbedder
from Code.Data.Graph.Embedders.graph_encoding import GraphEncoding
from Code.Data.Graph.context_graph import ContextGraph
from Code.Data.Text.data_sample import DataSample
from Code.Models.GNNs.gnn import GNN
from Code.Training import device
from Datasets.Batching.samplebatch import SampleBatch


class ContextGNN(GNN, ABC):

    """
    takes in context graphs as inputs, outputs graph encoding
    """

    def __init__(self, constructor: GraphConstructor, embedder: GraphEmbedder, gnnc: GNNConfig, configs: ConfigSet = None):
        GNN.__init__(self, None, gnnc, configs)
        self.embedder: GraphEmbedder = embedder
        self.constructor: GraphConstructor = constructor

        self.output_model = None

        self.layers = []
        self.layer_list: nn.Module = None

        self.last_batch_failures = None

    @property
    def gnnc(self):
        return self.configs.gnnc

    def init_model(self, data_sample: DataSample):
        encoding: GraphEncoding = self.get_graph_encoding(data_sample)
        in_features = encoding.x.size(-1)
        out_features = self.init_layers(in_features)
        self.sizes = [in_features, 1]

        self.layer_list = nn.ModuleList(self.layers).to(device)  # registers modules with pytorch and moves to device
        self.init_output_model(data_sample, out_features)

        # to initialise all sample dependant/ dynamically created params, before being passed to the optimiser
        self.forward(encoding)

    def init_output_model(self, data_sample: DataSample, in_features):
        # self.output_model = None
        # return
        out_type = data_sample.get_output_model()
        self.output_model = out_type(in_features).to(device)

    def forward(self, input: Union[ContextGraph, GraphEncoding, DataSample, SampleBatch]) -> GraphEncoding:
        """allows gnn to be used with either internal or external constructors and embedders"""
        data = self.get_graph_encoding(input)
        return self._forward(data)

    def get_graph_encoding(self, input: Union[ContextGraph, GraphEncoding, DataSample]) -> GraphEncoding:
        if isinstance(input, GraphEncoding):
            return input
        data = None
        if isinstance(input, SampleBatch):
            data: GraphEncoding = self.get_data_from_batch(input)
        if isinstance(input, ContextGraph):
            data: GraphEncoding = self.embedder(input)
        if isinstance(input, DataSample):
            data: GraphEncoding = self.get_data_from_data_sample(input)

        if not data:
            raise Exception()
        return data

    def get_data_from_batch(self, batch: SampleBatch) -> GraphEncoding:
        data_points = []
        self.last_batch_failures = []
        for bi in range(len(batch.batch_items)):
            batch_item = batch.batch_items[bi]
            try:
                data = self.get_data_from_data_sample(batch_item.data_sample, question=batch_item.question)
            except Exception as e:
                self.last_batch_failures.append(bi)
                continue
            # print("data:", data)
            data_points.append(data)
        if len(data_points) == 0:
            raise Exception("failed to create any valid data points from batch with " + repr(len(batch.batch_items))
                            + " items")
        batch = Batch.from_data_list(data_points)
        batch = GraphEncoding.from_geometric_batch(batch)
        # print("batch:", batch)
        return batch

    def get_data_from_data_sample(self, sample, question=None) -> GraphEncoding:
        graph = self.constructor(sample, question=question)
        data: GraphEncoding = self.embedder(graph)
        return data

    @abstractmethod
    def pass_layer(self, layer, data: GraphEncoding):
        # send the graph encoding through this gnn layer, ie call its forward
        pass

    def _forward(self, data: GraphEncoding) -> GraphEncoding:
        # format of graph layers forward: (x, edge_index, batch, **kwargs)
        # get x, autodetect feature count
        # init layers based on detected in_features and init args
        # print("cgnn operating on:",data,"\nx=",data.x)
        data = GNN._forward(self, data)  # add type and abs pos embeddings
        if self.layer_list is None:
            # gnn layers have not been initialised yet
            self.init_model(data.graph.data_sample)

        data.layer = 0
        next_layer = 0

        for layer in self.layers:
            next_layer = max(next_layer + 1, data.layer)
            data = self.pass_layer(layer, data)  # may or may not increase the layer count

            next_layer = max(next_layer, data.layer)
            data.layer = next_layer

            # print("layer",layer,"output:",data.x.size())

        if self.output_model:
            data = self.output_model(data)

        return data


