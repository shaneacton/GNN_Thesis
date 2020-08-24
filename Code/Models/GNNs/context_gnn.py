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


class ContextGNN(GNN):

    """
    takes in context graphs as inputs
    dynamically creates graph layers as needed based off edge types and state communications
    """

    """
    Plan: 
    get basic pipeline working with only current state updates, no special or previous/starting states
    add in support for previous/starting states - using state multiplexer
    move all encoder params into the CG-GNN with options to fine tune or not
    save/loading of cg-gnn with encoders
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
        self.layers: List[GraphModule] = []
        self.layer_list = None

    @property
    def gnnc(self):
        return self.configs.gnnc

    def init_layers(self, in_features):
        """creates layers based on the gnn config provided as well as the sampled in features size"""
        for layer_conf in self.gnnc.layers:
            layer_features = layer_conf[gnn_config.NUM_FEATURES]

            layer = GraphModule([in_features, layer_features, layer_features], layer_conf)
            self.layers.append(layer)
            in_features = layer_features

        out_type = self.gnnc.output_layer[gnn_config.LAYER_TYPE]
        self.layers.append(out_type(in_features))
        print("adding output layer:",self.layers[-1])
        self.layer_list = nn.ModuleList(self.layers).to(device)  # registers modules with pytorch and moves to device

    def forward(self, input: Union[ContextGraph, GraphEncoding, DataSample]) -> GraphEncoding:
        """allows gnn to be used with either internal or external constructors and embedders"""
        if isinstance(input, GraphEncoding):
            return self._forward(input)
        data = None
        if isinstance(input, ContextGraph):
            data: GraphEncoding = self.embedder(input)
        if isinstance(input, DataSample):
            graph = self.constructor(input)
            data: GraphEncoding = self.embedder(graph)
        if not data:
            raise Exception()
        return self._forward(data)

    def _forward(self, data: GraphEncoding) -> GraphEncoding:
        # format of graph layers forward: (x, edge_index, batch, **kwargs)
        # get x, autodetect feature count
        # init layers based on detected in_features and init args
        # print("cgnn operating on:",data,"\nx=",data.x)
        if self.layer_list is None:
            in_features = data.x[0].size(-1)
            self.init_layers(in_features)

        for layer in self.layers:
            # try:
            data = layer(data)
            # except Exception as e:
            #     print("failed to prop through", layer, "with kwargs:", kwargs)
            #     raise e

            # print("layer",layer,"output:",out)

        return data
