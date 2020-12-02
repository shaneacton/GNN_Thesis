from abc import ABC, abstractmethod
from typing import Union, Dict, List

import torch
from torch import nn, Tensor
from torch_geometric.data import Batch
from transformers import LongformerConfig
from transformers.modeling_longformer import LongformerPreTrainedModel

from Code.Config import GNNConfig
from Code.Config.config_set import ConfigSet
from Code.Data.Graph.Contructors.qa_graph_constructor import QAGraphConstructor, TooManyEdgesException
from Code.Data.Graph.Embedders.graph_embedder import GraphEmbedder
from Code.Data.Graph.graph_encoding import GraphEncoding
from Code.Data.Graph.context_graph import QAGraph
from Code.Data.Text.text_utils import question, candidates, question_key, has_candidates, num_candidates
from Code.Models.GNNs.OutputModules.output_model import OutputModel

from Code.Models.GNNs.gnn import GNN
from Code.Models.Loss.loss_funcs import get_span_element_loss, get_span_loss
from Code.Models.context_nn import ContextNN
from Code.Play.initialiser import get_longformer_config
from Code.Training import device
from Code.constants import CONTEXT
try:
    from Viz.context_graph_visualiser import render_graph
except:  # graph viz not installed
    pass

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

        self.output_model: OutputModel = None

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
        print("initialising context gnn with", self.output_model)

    def forward(self, input: Union[QAGraph, GraphEncoding, Dict], **kwargs) -> GraphEncoding:
        """allows gnn to be used with either internal or external constructors and embedders"""
        if isinstance(input, Dict):
            """moves all non essential inputs into kwargs"""
            input, kwargs = prep_input(input, kwargs)
        try:
            data = self.get_graph_encoding(input)
        except TooManyEdgesException as e:
            if not self.output_model:
                print("cannot recover from too many edges. First example must pass")
                raise e
            has_answers = 'answer' in kwargs or 'start_positions' in kwargs
            # print("returning null output")
            return self.get_null_return(num_candidates(input), input, has_answers)

        if data.is_batched:
            raise Exception("batched data not supported yet")

        return self._forward(data, **kwargs)

    def get_null_return(self, num_outputs, example, include_loss:bool):
        loss = None
        if include_loss:
            loss = torch.tensor(0., requires_grad=True)
        logits = torch.tensor([0.] * num_outputs, requires_grad=True).to(float)
        if has_candidates(example):
            output = logits
        else:
            output = (logits, logits)
        return ((loss,) + (output,)) if loss is not None else output

    def get_graph_encoding(self, input: Union[QAGraph, GraphEncoding, Dict]) -> GraphEncoding:
        """graph encoding is done with a batchsize of 1"""
        if isinstance(input, GraphEncoding):
            return input
        data = None
        if isinstance(input, QAGraph):
            raise Exception()
            # data: GraphEncoding = self.embedder(input)
        if isinstance(input, Dict):
            data: GraphEncoding = self.get_graph_from_data_sample(input)

        if not data:
            raise Exception()
        return data

    def get_graph_from_data_sample(self, example) -> GraphEncoding:
        graphs: Union[List[QAGraph], QAGraph] = self.constructor(example)

        if isinstance(graphs, List):
            """graph batching"""
            # todo this graph embedding can be done in parallel
            datas: List[GraphEncoding] = [self.embedder(graph) for graph in graphs]
            data = GraphEncoding.batch(datas)
        else:
            data: GraphEncoding = self.embedder(graphs)
        # render_graph(graphs[0] if isinstance(graphs, List) else graphs, self.embedder.text_encoder)

        return data

    @abstractmethod
    def pass_layer(self, layer, data: GraphEncoding):
        # send the graph encoding through this gnn layer, ie call its forward
        pass

    def _forward(self, data: GraphEncoding, **kwargs) -> Union[Tensor, GraphEncoding]:
        """returns the transformed context graph encoding, and the loss"""
        # format of graph layers forward: (x, edge_index, batch, **kwargs)
        # get x, autodetect feature count
        # init layers based on detected in_features and init args
        if isinstance(data, List):
            raise Exception("graph encodings should be batched via pytorch geometric batch class")
        if self.layer_list is None:
            # gnn layers have not been initialised yet
            self.init_model(data.graphs[0].example)
        data = GNN._forward(self, data)  # add type and abs pos embeddings

        for layer in self.layers:
            data = self.pass_layer(layer, data)

        kwargs.update({"source": CONTEXT})
        out = self.output_model(data, **kwargs)
        # print("out:", out)

        return out


