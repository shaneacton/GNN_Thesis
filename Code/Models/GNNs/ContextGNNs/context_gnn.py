from abc import ABC, abstractmethod
from typing import Union, Dict

from torch import nn
from torch_geometric.data import Batch

import Code.Data.Text.text_utils
from Code.Config import GNNConfig
from Code.Config.config_set import ConfigSet
from Code.Data.Graph.Contructors.graph_constructor import GraphConstructor
from Code.Data.Graph.Embedders.graph_embedder import GraphEmbedder
from Code.Data.Graph.Embedders.graph_encoding import GraphEncoding
from Code.Data.Graph.context_graph import QAGraph

from Code.Models.GNNs.OutputModules.candidate_selection import CandidateSelection
from Code.Models.GNNs.gnn import GNN
from Code.Models.context_nn import ContextNN
from Code.Training import device


class ContextGNN(GNN, ContextNN, ABC):

    """
    takes in context graphs as inputs, outputs graph encoding
    """

    def __init__(self, constructor: GraphConstructor, embedder: GraphEmbedder, gnnc: GNNConfig, configs: ConfigSet = None):
        GNN.__init__(self, None, gnnc, configs)
        ContextNN.__init__(self)
        self.embedder: GraphEmbedder = embedder
        self.constructor: GraphConstructor = constructor

        self.output_model = None

        self.layers = []
        self.layer_list: nn.Module = None

        self.last_batch_failures = None

    @property
    def gnnc(self):
        return self.configs.gnnc

    def init_model(self, example):
        encoding: GraphEncoding = self.get_graph_encoding(data_sample)
        in_features = encoding.x.size(-1)
        out_features = self.init_layers(in_features)
        self.sizes = [in_features, 1]

        self.layer_list = nn.ModuleList(self.layers).to(device)  # registers modules with pytorch and moves to device
        self.init_output_model(data_sample, out_features)

        # to initialise all sample dependant/ dynamically created params, before being passed to the optimiser
        self.forward(encoding)

    def forward(self, input: Union[QAGraph, GraphEncoding, DataSample, SampleBatch]) -> GraphEncoding:
        """allows gnn to be used with either internal or external constructors and embedders"""
        data = self.get_graph_encoding(input)
        return self._forward(data)

    def get_graph_encoding(self, input: Union[QAGraph, GraphEncoding, Dict]) -> GraphEncoding:
        """graph encoding is done with a batchsize of 1"""
        if isinstance(input, GraphEncoding):
            return input
        data = None
        if isinstance(input, QAGraph):
            data: GraphEncoding = self.embedder(input)
        if isinstance(input, Dict):
            data: GraphEncoding = self.get_graph_from_data_sample(input)

        if not data:
            raise Exception()
        return data

    def get_graph_from_data_sample(self, example) -> GraphEncoding:
        graph = self.constructor(example)
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
            self.init_model(data.graph.example)

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

    def get_output_model_type(self, example: Dict):
        # answer_type = data_sample.get_answer_type()
        # if answer_type == CandidateAnswer:
        #     return CandidateSelection
        # return ContextNN.get_output_model_type(self, data_sample)
        pass


