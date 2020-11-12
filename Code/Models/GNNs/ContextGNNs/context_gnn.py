from abc import ABC, abstractmethod
from typing import Union, Dict, List

import torch
from torch import nn
from torch_geometric.data import Batch

from Code.Config import GNNConfig
from Code.Config.config_set import ConfigSet
from Code.Data.Graph.Contructors.qa_graph_constructor import QAGraphConstructor
from Code.Data.Graph.Embedders.graph_embedder import GraphEmbedder
from Code.Data.Graph.graph_encoding import GraphEncoding
from Code.Data.Graph.context_graph import QAGraph

from Code.Models.GNNs.gnn import GNN
from Code.Models.context_nn import ContextNN
from Code.Training import device


def prep_input(input, kwargs):
    """moves all non essential fields from in to kwargs"""
    input_fields = ["context", "question"]
    graph_input = {}
    for inp in input:
        if inp in input_fields:
            graph_input[inp] = input[inp]
        else:
            kwargs[inp] = input[inp]

    return graph_input, kwargs


class ContextGNN(GNN, ContextNN, ABC):

    """
    takes in context graphs as inputs, outputs graph encoding
    """

    def __init__(self, embedder: GraphEmbedder, gnnc: GNNConfig, configs: ConfigSet = None):
        GNN.__init__(self, None, gnnc, configs)
        ContextNN.__init__(self)
        self.embedder: GraphEmbedder = embedder
        self.constructor: QAGraphConstructor = QAGraphConstructor(embedder.gcc)

        self.output_model = None

        self.layers = []
        self.layer_list: nn.Module = None

        self.last_batch_failures = None

    @property
    def gnnc(self):
        return self.configs.gnnc

    def init_model(self, example):
        encoding: GraphEncoding = self.get_graph_encoding(example)
        in_features = encoding.x.size(-1)
        out_features = self.init_layers(in_features)
        self.sizes = [in_features, 1]

        self.layer_list = nn.ModuleList(self.layers).to(device)  # registers modules with pytorch and moves to device
        self.init_output_model(example, out_features)

        # to initialise all sample dependant/ dynamically created params, before being passed to the optimiser
        self.forward(encoding)

    def forward(self, input: Union[QAGraph, GraphEncoding, Dict], **kwargs) -> GraphEncoding:
        """allows gnn to be used with either internal or external constructors and embedders"""
        if isinstance(input, Dict):
            # graph still needs to be constructed
            # print("kwargs:", kwargs, "input:", input)
            input, kwargs = prep_input(input, kwargs)
            # print("after k:", kwargs, "\nin:", input)
        data = self.get_graph_encoding(input)

        # if data.is_batched:
        #     print("passing batched graph", data, "through gnn")
        #     print("e_s:", data.edge_index.size())
        #     print("edge:", data.edge_index)
        #     print("edge min:", torch.min(data.edge_index), "max:", torch.max(data.edge_index))
        return self._forward(data, **kwargs)

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
        graphs: Union[List[QAGraph], QAGraph] = self.constructor(example)
        if isinstance(graphs, List):
            # todo this graph embedding can be done in parallel
            datas: List[GraphEncoding] = [self.embedder(graph) for graph in graphs]
            data = GraphEncoding.batch(datas)
        else:
            data: GraphEncoding = self.embedder(graphs)
        return data

    @abstractmethod
    def pass_layer(self, layer, data: GraphEncoding):
        # send the graph encoding through this gnn layer, ie call its forward
        pass

    def _forward(self, data: GraphEncoding, **kwargs) -> GraphEncoding:
        # format of graph layers forward: (x, edge_index, batch, **kwargs)
        # get x, autodetect feature count
        # init layers based on detected in_features and init args
        if isinstance(data, List):
            raise Exception("graph encodings should be batched via pytorch geometric batch class")

        if self.layer_list is None:
            # gnn layers have not been initialised yet
            self.init_model(data.graphs[0].example)
        data = GNN._forward(self, data)  # add type and abs pos embeddings

        data.layer = 0
        next_layer = 0

        for layer in self.layers:
            next_layer = max(next_layer + 1, data.layer)
            data = self.pass_layer(layer, data)  # may or may not increase the layer count
            next_layer = max(next_layer, data.layer)
            data.layer = next_layer

            # print("layer",layer,"output:",data.x.size())

        if self.output_model:
            data = self.output_model(data, **kwargs)

        return data


