from typing import List, Type

import torch
from torch_geometric.nn import TopKPooling, SAGEConv, MessagePassing

from Code.Data import Graph
from Code.Data.Graph.Embedders.graph_encoding import GraphEncoding
from Code.Models.GNNs.Layers.graph_layer import GraphLayer
from Code.Models.GNNs.Layers.CustomLayers.prop_and_pool_layer import PropAndPoolLayer
from Code.Training import device

from torch import nn


class GraphModule(GraphLayer):

    """
    a structure to repeat any graph_layer which takes in_size and out_size args

    creates the minimum amount of layers needed to convert the num features given
    from input_size->hidden_size->output_size with the specified number of hidden layers

    a module can have no hidden layers: meaning only in_layer(in_size->hid_size) and out_layer(hid_size,out_size)
    a module needs no output layer if hid_size==out_size
    a module needs no input layer if there are hidden layers, and the in_size==hidden_size
    """

    def __init__(self, sizes: List[int], layer_type: Type[MessagePassing], distinct_weight_repeats, same_weight_repeats=1,
                 repeated_layer_args=None, return_all_outputs=False):
        """
        :param sizes: [in_size, hidden_size, out_size]
        :param distinct_weight_repeats: number of unique hidden layers which each get their own weight params
        :param same_weight_repeats: number of times to recurrently pass through the hidden layers.
        Increasing this increases the number of layers the input passes through without increasing
        the trainable num params
        """

        self.repeated_layer_args = repeated_layer_args if repeated_layer_args else {}
        self.same_weight_repeats = same_weight_repeats
        self.distinct_weight_repeats = distinct_weight_repeats
        self.repeated_layer_type = layer_type
        self.return_all_outputs = return_all_outputs
        if distinct_weight_repeats and not same_weight_repeats:
            raise Exception()
        if len(sizes) != 3:
            raise Exception("please provide input,hidden,output sizes")
        super().__init__(sizes)

        self.module = self.initialise_module()

    @property
    def hidden_size(self):
        return self.sizes[1]

    def get_layer(self):
        for layer in self.layer:
            return layer

    def initialise_module(self):
        has_hidden = self.distinct_weight_repeats > 0

        needs_output = self.hidden_size != self.output_size

        ommit_input = has_hidden and self.input_size == self.hidden_size
        needs_input = not ommit_input

        def new_layer(in_size, out_size):
            sizes = [in_size, out_size]
            if issubclass(self.repeated_layer_type, GraphLayer):
                return self.repeated_layer_type(sizes, **self.repeated_layer_args)
            else:
                activation_type=None
                if "activation_type" in self.repeated_layer_args:
                    activation_type = self.repeated_layer_args.pop("activation_type")

                # return GraphLayer(sizes, self.repeated_layer_type, activation_type=activation_type,
                #                   layer_args=self.repeated_layer_args)
                return GraphLayer(sizes)

        layers = [new_layer(self.input_size, self.hidden_size)] if needs_input else []
        layers += [new_layer(self.hidden_size, self.hidden_size) for _ in range(self.distinct_weight_repeats)] * self.same_weight_repeats
        layers += [new_layer(self.hidden_size, self.output_size)] if needs_output else []

        return nn.Sequential(*layers)

    def forward(self, data: GraphEncoding):
        all_graph_states = []
        for layer in self.module:
            """passes x through each item in the seq block and optionally records intermediate outputs"""

            data = layer(data)
            x = data.x
            if self.return_all_outputs:
                all_graph_states.append(x)

        if self.return_all_outputs:
            # print("returning:",(x, all_graph_states))
            return data, all_graph_states
        return data


if __name__ == "__main__":
    torch.manual_seed(0)

    pnp_args = {"prop_type": SAGEConv, "pool_type": TopKPooling, "pool_args": {"ratio":0.8}}
    pnp = GraphModule([3, 128, 128], PropAndPoolLayer, 2, same_weight_repeats=1, repeated_layer_args=pnp_args,
                      return_all_outputs=True).to(device)
    print(pnp)
    print(pnp.num_params)

    out = pnp(Graph.example_batch.x, Graph.example_batch.edge_index, Graph.example_batch.batch)
    x, outs = out

    print(x)