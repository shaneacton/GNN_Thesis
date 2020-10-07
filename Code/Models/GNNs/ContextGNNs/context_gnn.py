import time
from abc import ABC, abstractmethod
from typing import List, Union

from torch import nn

from Code.Config import GNNConfig, gnn_config
from Code.Config.config_set import ConfigSet
from Code.Data.Graph.Contructors.graph_constructor import GraphConstructor
from Code.Data.Graph.Embedders.graph_embedder import GraphEmbedder
from Code.Data.Graph.Embedders.graph_encoding import GraphEncoding
from Code.Data.Graph.context_graph import ContextGraph
from Code.Data.Text.data_sample import DataSample
from Code.Models.GNNs.gnn import GNN
from Code.Models.GNNs.graph_module import GraphModule
from Code.Training import device


class ContextGNN(GNN, ABC):

    """
    takes in context graphs as inputs, outputs graph encoding
    """

    def __init__(self, constructor: GraphConstructor, embedder: GraphEmbedder, gnnc: GNNConfig, configs: ConfigSet = None):
        super().__init__()
        self.constructor: GraphConstructor = constructor
        self.configs: ConfigSet = configs
        if not self.configs:
            self.configs = ConfigSet(config=gnnc)
        elif not self.configs.gnnc:
            self.configs.add_config(gnnc)

        self.embedder: GraphEmbedder = embedder

        self.output_model = None

        self.layers = []
        self.layer_list: nn.Module = None

    @property
    def gnnc(self):
        return self.configs.gnnc

    def init_model(self, data_sample: DataSample):
        encoding: GraphEncoding = self.get_graph_encoding(data_sample)
        in_features = encoding.x.size(-1)

        out_features = self.init_layers(in_features)
        self.layer_list = nn.ModuleList(self.layers).to(device)  # registers modules with pytorch and moves to device
        self.init_output_model(data_sample, out_features)

        # to initialise all sample dependant/ dynamically created params, before being passed to the optimiser
        self.forward(encoding)

    @abstractmethod
    def init_layers(self, in_features) -> int:  # returns the feature num of the last layer
        pass

    def init_output_model(self, data_sample: DataSample, in_features):
        # self.output_model = None
        # return
        out_type = data_sample.get_output_model()
        self.output_model = out_type(in_features).to(device)

    def forward(self, input: Union[ContextGraph, GraphEncoding, DataSample]) -> GraphEncoding:
        """allows gnn to be used with either internal or external constructors and embedders"""
        return self._forward(self.get_graph_encoding(input))

    def get_graph_encoding(self, input: Union[ContextGraph, GraphEncoding, DataSample]) -> GraphEncoding:
        if isinstance(input, GraphEncoding):
            return input
        data = None
        if isinstance(input, ContextGraph):
            data: GraphEncoding = self.embedder(input)
        if isinstance(input, DataSample):
            construction_start_time = time.time()
            graph = self.constructor(input)
            encoding_start_time = time.time()
            construction_time = encoding_start_time - construction_start_time
            data: GraphEncoding = self.embedder(graph)
            encoding_time = time.time() - encoding_start_time

            # print("construction time:", construction_time, "embedding time:",encoding_time, "total:",
            #       (construction_time + encoding_time))

        if not data:
            raise Exception()
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


