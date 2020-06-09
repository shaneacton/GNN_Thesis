import inspect
from typing import List, Type, Dict, Any

from torch import nn, Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops


class GraphLayer(MessagePassing):
    """wrapper around a propagation layer such as SAGEConv, GATConv etc"""

    def __init__(self, sizes: List[int], layer_type:Type[MessagePassing], activation_type=None, layer_args=None):
        super().__init__()
        self.layer_type = layer_type
        self.sizes = sizes
        self.layer_args = layer_args if layer_args else {}

        self.layer = self.initialise_layer()
        self.activation = activation_type() if activation_type else None

    @property
    def input_size(self):
        return self.sizes[0]

    @property
    def output_size(self):
        return self.sizes[-1]

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_layer(self):
        return self.layer

    def initialise_layer(self):
        if issubclass(self.layer_type, GraphLayer):
            raise Exception("Base Graph layer must contain a GNN primitive layer")

        print("creating layer", self.layer_type,"sizes:",self.sizes, "args:",self.layer_args)
        return self.layer_type(*self.sizes, **self.layer_args)

    def get_base_layer_all_arg_names(self):
        """gets all the arguments that this layer would need throughout its prop"""
        all = self.get_base_layer_update_arg_names() + self.get_base_layer_forward_arg_names()
        all += self.get_base_layer_message_arg_names()
        all += ["edge_index", "edge_types", "batch"]
        return list(set(all))  # returns a list of unique args which are used through this layers propagation

    def get_base_layer_message_arg_names(self):
        return inspect.getfullargspec(self.get_layer().message)[0]

    def get_base_layer_update_arg_names(self):
        return inspect.getfullargspec(self.get_layer().update)[0]

    def get_base_layer_forward_arg_names(self):
        return inspect.getfullargspec(self.get_layer().forward)[0]

    def get_base_layer_forward_defaults(self):
        """
        gets the defaulted values from the forward method
        these defaulted args often become required when moving from forward to message
        thus they must be passed explicitly into message if needed
        """
        all_args = self.get_base_layer_forward_arg_names()
        default_values = inspect.getfullargspec(self.layer.forward)[3]
        defaulted_args = all_args[len(all_args) - len(default_values):]
        return {defaulted_args[i]: default_values[i] for i in range(len(defaulted_args))}

    def get_kwargs_with_defaults(self, kwargs):
        """fuses the given kwargs with the defaulted values from the forward method"""
        defaults = self.get_base_layer_forward_defaults()
        defaults.update(kwargs)  # will override defaults which are provided
        return defaults

    def get_needed_args(self, accepted_args, available_args):
        """returns all of the available args which are accepted"""
        return {arg: available_args[arg] for arg in available_args.keys() if arg in accepted_args}

    def message(self, x_j: Tensor, kwargs: Dict[str, Any]):
        needed_args = self.get_needed_args(self.get_base_layer_message_arg_names(), kwargs)
        return self.layer.message(x_j, **needed_args)

    def update(self, inputs, kwargs: Dict[str, Any]):
        needed_args = self.get_needed_args(self.get_base_layer_update_arg_names(), kwargs)
        return self.layer.update(inputs, **needed_args)

    def forward(self, x, edge_index, batch, **kwargs):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        kwargs = self.get_kwargs_with_defaults(kwargs)
        # print("kwargs with defaults:",kwargs)

        kwargs.update({"batch":batch, "x":x})
        print("propping args:",kwargs)
        x = self.propagate(edge_index, x=x, kwargs=kwargs)
        # x = self.layer(x, edge_index)
        if self.activation:
            return self.activation(x)
        return x
