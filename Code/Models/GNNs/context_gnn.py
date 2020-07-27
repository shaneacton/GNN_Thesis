from typing import List

from torch import nn

from Code.Config import GNNConfig, gnn_config
from Code.Data.Graph.Embedders.graph_embedder import GraphEmbedder
from Code.Data.Graph.Embedders.graph_encoding import GraphEncoding
from Code.Data.Graph.context_graph import ContextGraph
from Code.Models.GNNs.Abstract.gnn import GNN
from Code.Models.GNNs.Abstract.graph_module import GraphModule
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

    def __init__(self, embedder: GraphEmbedder, gnnc: GNNConfig):
        super().__init__()
        self.gnnc = gnnc
        self.embedder: GraphEmbedder = embedder
        self.layers: List[GraphModule] = []
        self.layer_list = None

    def init_layers(self, in_features):
        """creates layers based on the gnn config provided as well as the sampled in features size"""
        for layer_conf in self.gnnc.layers:
            layer_features = layer_conf[gnn_config.NUM_FEATURES]

            layer = GraphModule([in_features, layer_features, layer_features], layer_conf[gnn_config.LAYER_TYPE],
                                layer_conf[gnn_config.DISTINCT_WEIGHT_REPEATS],
                                num_hidden_repeats=layer_conf[gnn_config.SAME_WEIGHT_REPEATS],
                                repeated_layer_args=layer_conf[gnn_config.LAYER_ARGS])
            self.layers.append(layer)
            in_features = layer_features
        self.layer_list = nn.ModuleList(self.layers).to(device)  # registers modules with pytorch and moves to device

    def forward(self, graph: ContextGraph):
        # format of graph layers forward: (x, edge_index, batch, **kwargs)
        # get x, autodetect feature count
        # init layers based on detected in_features and init args
        data: GraphEncoding = self.embedder(graph)
        print("cgnn operating on:",data,"\nx=",data.x)
        if self.layer_list is None:
            in_features = data.x[0].size(-1)
            self.init_layers(in_features)

        for layer in self.layers:
            kwargs = self.get_required_kwargs_from_batch(data, layer)

            # try:
            out = layer(**kwargs)
            # except Exception as e:
            #     print("failed to prop through", layer, "with kwargs:", kwargs)
            #     raise e

            # print("layer",layer,"output:",out)

        return out
