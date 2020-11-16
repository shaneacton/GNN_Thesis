from abc import ABC, abstractmethod
from typing import Union, Dict, List

import torch
from torch import nn
from torch_geometric.data import Batch
from transformers import LongformerConfig
from transformers.modeling_longformer import LongformerPreTrainedModel

from Code.Config import GNNConfig
from Code.Config.config_set import ConfigSet
from Code.Data.Graph.Contructors.qa_graph_constructor import QAGraphConstructor
from Code.Data.Graph.Embedders.graph_embedder import GraphEmbedder
from Code.Data.Graph.graph_encoding import GraphEncoding
from Code.Data.Graph.context_graph import QAGraph
from Code.Data.Text.text_utils import question, candidates, question_key

from Code.Models.GNNs.gnn import GNN
from Code.Models.context_nn import ContextNN
from Code.Play.initialiser import get_longformer_config
from Code.Training import device
from Code.constants import CONTEXT
# from Viz.context_graph_visualiser import render_graph


def prep_input(input, kwargs):
    """moves all non essential fields from in to kwargs"""
    input_fields = ["context", question_key(input)]
    if candidates(input):
        input_fields.append("candidates")
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

    def __init__(self, embedder: GraphEmbedder, gnnc: GNNConfig, configs: ConfigSet = None, longformer_config: LongformerConfig=None):
        GNN.__init__(self, None, gnnc, configs)
        ContextNN.__init__(self)
        if longformer_config is None:
            longformer_config = get_longformer_config()
        self.config:LongformerConfig = longformer_config
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
        # self.forward(encoding)

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
            example = graphs[0].example
        else:
            data: GraphEncoding = self.embedder(graphs)
            example = graphs.example
        # ctx_enc = self.embedder.text_encoder.get_context_encoding(example)
        # q_enc = self.embedder.text_encoder.get_question_encoding(example)
        # render_graph(graphs[0] if isinstance(graphs, List) else graphs, ctx_enc, q_enc)

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
        # inited = False
        if self.layer_list is None:
            # gnn layers have not been initialised yet
            self.init_model(data.graphs[0].example)
            inited = True
        data = GNN._forward(self, data)  # add type and abs pos embeddings

        data.layer = 0
        next_layer = 0

        for layer in self.layers:
            next_layer = max(next_layer + 1, data.layer)
            data = self.pass_layer(layer, data)  # may or may not increase the layer count
            next_layer = max(next_layer, data.layer)
            data.layer = next_layer

            # print("layer",layer,"output:",data.x.size())

        kwargs.update({"source": CONTEXT})
        if self.output_model:
            out = self.output_model(data, **kwargs)
        #     print("out:", type(out), out)
        #
        # print("out shape:", out[1].size())
        # print("loss:", out[0])
        # if inited:
        #     raise Exception("weh")
        return out


