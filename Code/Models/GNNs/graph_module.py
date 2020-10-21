from typing import List

from torch import nn

import Code.constants
from Code.Config import gnn_config, GNNConfig
from Code.Data.Graph.Embedders.graph_encoding import GraphEncoding
from Code.Models.GNNs.Layers.layer_constructor import LayerConstructor
from Code.Models.GNNs.gnn_component import GNNComponent


class GraphModule(GNNComponent, nn.Module):

    """
    a structure to repeat any graph_layer which takes in_size and out_size args

    creates the minimum amount of layers needed to convert the num features given
    from input_size->hidden_size->output_size with the specified number of hidden layers

    a module can have no hidden layers: meaning only in_layer(in_size->hid_size) and out_layer(hid_size,out_size)
    a module needs no output layer if hid_size==out_size
    a module needs no input layer if there are hidden layers, and the in_size==hidden_size
    """

    def __init__(self, sizes: List[int], layer_conf, gnnc: GNNConfig, activation_type=None, dropout_ratio=None, activation_kwargs=None):
        """
        :param sizes: [in_size, hidden_size, out_size]
        Increasing this increases the number of layers the input passes through without increasing
        the trainable num params
        """

        self.gnnc = gnnc
        self.same_weight_repeats = layer_conf[Code.constants.SAME_WEIGHT_REPEATS]
        self.distinct_weight_repeats = layer_conf[Code.constants.DISTINCT_WEIGHT_REPEATS]
        self.return_all_outputs = False
        if self.distinct_weight_repeats and not self.same_weight_repeats:
            raise Exception()
        if len(sizes) != 3:
            raise Exception("please provide input,hidden,output sizes")
        self.layer_conf = layer_conf

        nn.Module.__init__(self)
        GNNComponent.__init__(self, sizes, activation_type, dropout_ratio, activation_kwargs=activation_kwargs)

        self.module = self.initialise_module()

    def get_layer(self):
        for layer in self.layer:
            return layer

    def initialise_module(self):
        has_hidden = self.distinct_weight_repeats > 0

        needs_output = self.hidden_size != self.output_size

        ommit_input = has_hidden and self.input_size == self.hidden_size
        needs_input = not ommit_input

        layer_constructor = LayerConstructor()

        def new_layer(in_size, out_size):
            sizes = [in_size, out_size]
            return layer_constructor.get_layer(sizes, self.layer_conf, self.gnnc.global_params)

        layers = [new_layer(self.input_size, self.hidden_size)] if needs_input else []
        layers += [new_layer(self.hidden_size, self.hidden_size) for _ in range(self.distinct_weight_repeats)] * self.same_weight_repeats
        layers += [new_layer(self.hidden_size, self.output_size)] if needs_output else []

        return nn.Sequential(*layers)

    def forward(self, data: GraphEncoding):
        all_graph_states = []
        for layer in self.module:
            """passes x through each item in the seq block and optionally records intermediate outputs"""
            next_layer = data.layer + 1

            data = layer(data)  # may or may not increase the layer
            x = data.x

            next_layer = max(next_layer, data.layer)
            data.layer = next_layer

            if self.return_all_outputs:
                all_graph_states.append(x)

        if self.return_all_outputs:
            # print("returning:",(x, all_graph_states))
            return data, all_graph_states
        return data
