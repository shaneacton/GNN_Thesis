from typing import List, Type

from torch import nn
from torch_geometric.nn import TopKPooling, SAGEConv, MessagePassing

from Code.Models.GNNs.graph_layer import GraphLayer
from Code.Models.GNNs.prop_and_pool_layer import PropAndPoolLayer


class GraphModule(GraphLayer):

    """
    a structure to repeat any graph_layer which takes in_size and out_size args

    creates the minimum amount of layers needed to convert the num features given
    from input_size->hidden_size->output_size with the specified number of hidden layers

    a module can have no hidden layers: meaning only in_layer(in_size->hid_size) and out_layer(hid_size,out_size)
    a module needs no output layer if hid_size==out_size
    a module needs no input layer if there are hidden layers, and the in_size==hidden_size
    """

    def __init__(self, sizes: List[int], layer_type: Type[MessagePassing], num_hidden_layers, num_hidden_repeats=1,
                 repeated_layer_args=None):
        """
        :param sizes: [in_size, hidden_size, out_size]
        :param num_hidden_layers: number of unique hidden layers which each get their own weight params
        :param num_hidden_repeats: number of times to recurrently pass through the hidden layers.
        Increasing this increases the number of layers the input passes through without increasing
        the trainable num params
        """

        self.repeated_layer_args = repeated_layer_args if repeated_layer_args else {}
        self.num_hidden_repeats = num_hidden_repeats
        self.num_hidden_layers = num_hidden_layers
        self.repeated_layer_type = layer_type
        if num_hidden_layers and not num_hidden_repeats:
            raise Exception()
        if len(sizes) != 3:
            raise Exception("please provide input,hidden,output sizes")
        super().__init__(sizes, GraphModule)

    @property
    def hidden_size(self):
        return self.sizes[1]

    def initialise_layer(self):
        has_hidden = self.num_hidden_layers > 0

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

                return GraphLayer(sizes, self.repeated_layer_type, activation_type=activation_type,
                                  layer_args=self.repeated_layer_args)

        layers = [new_layer(self.input_size, self.hidden_size)] if needs_input else []
        layers += [new_layer(self.hidden_size, self.hidden_size) for i in range(self.num_hidden_layers)] * self.num_hidden_repeats
        layers += [new_layer(self.hidden_size, self.output_size)] if needs_output else []

        return nn.Sequential(*layers)


if __name__ == "__main__":
    pnp_args = {"activation_type":nn.ReLU, "prop_type": SAGEConv, "pool_type": TopKPooling, "pool_args": {"ratio":0.8}}
    pnp = GraphModule([1000, 1000, 500], PropAndPoolLayer, 2, num_hidden_repeats=1, repeated_layer_args=pnp_args)
    print(pnp)
    print(pnp.num_params)
    pnp = GraphModule([1500, 1000, 1000], PropAndPoolLayer, 0, num_hidden_repeats=2, repeated_layer_args=pnp_args)
    print(pnp)
    print(pnp.num_params)
