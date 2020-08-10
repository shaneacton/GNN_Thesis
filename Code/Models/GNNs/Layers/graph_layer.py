import inspect
from typing import List, Type, Dict, Any

from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops

from Code.Data.Graph.Embedders.graph_encoding import GraphEncoding


class GraphLayer(MessagePassing):
    """wrapper around a propagation layer such as SAGEConv, GATConv etc"""

    def __init__(self, sizes: List[int], layer_type:Type[MessagePassing], activation_type=None, layer_args=None, init_layer=True):
        super().__init__()
        self.layer_type = layer_type
        self.sizes = sizes
        self.layer_args = layer_args if layer_args else {}

        if init_layer:
            self.layer = self.initialise_layer()
        else:
            self.layer = None
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
        default_values = inspect.getfullargspec(self.get_layer().forward)[3]
        if default_values is None:
            return {}

        defaulted_args = all_args[len(all_args) - len(default_values):]
        return {defaulted_args[i]: default_values[i] for i in range(len(defaulted_args))}

    def get_kwargs_with_defaults(self, kwargs):
        """fuses the given kwargs with the defaulted values from the forward method"""
        defaults = self.get_base_layer_forward_defaults()
        defaults.update(kwargs)  # will override defaults which are provided
        return defaults

    @staticmethod
    def get_needed_args(accepted_args, available_args):
        """returns all of the available args which are accepted"""
        return {arg: available_args[arg] for arg in available_args.keys() if arg in accepted_args}

    def message(self, x_j: Tensor, kwargs: Dict[str, Any]):
        needed_args = self.get_needed_args(self.get_base_layer_message_arg_names(), kwargs)
        return self.layer.message(x_j, **needed_args)

    def update(self, inputs, kwargs: Dict[str, Any]):
        needed_args = self.get_needed_args(self.get_base_layer_update_arg_names(), kwargs)
        return self.layer.update(inputs, **needed_args)

    def get_required_kwargs_from_batch(self, data: GraphEncoding):
        """
        return only the params of the data object which are needed by the given layer
        """
        layer_args = self.get_base_layer_all_arg_names()  # needed
        data_args = data.__dict__  # given
        present_args = {arg: data_args[arg] for arg in layer_args if arg in data_args.keys()}
        return present_args

    def prepare_args(self, data: GraphEncoding):
        kwargs = self.get_required_kwargs_from_batch(data)
        edge_index = self.clean_loops(kwargs, data.x.size(0))
        kwargs = self.get_kwargs_with_defaults(kwargs)
        return edge_index, kwargs

    def forward(self, data: GraphEncoding):
        edge_index, kwargs = self.prepare_args(data)

        print("propping args:", kwargs)
        x = self.propagate(edge_index, x=data.x, kwargs=kwargs)
        if self.activation:
            x = self.activation(x)
        data.x = x
        return data

    @staticmethod
    def clean_loops(kwargs, num_nodes):
        edge_types = kwargs["edge_types"] if "edge_types" in kwargs else None
        edge_index = kwargs["edge_index"]

        try:
            edge_index, edge_types = remove_self_loops(edge_index, edge_attr=edge_types)
        except Exception as e:
            print("error cleaning loops in edge indiex:",edge_index, "kwargs:",kwargs)
            raise e
        edge_index, edge_types = add_self_loops(edge_index, num_nodes=num_nodes, edge_weight=edge_types)
        kwargs["edge_types"] = edge_types
        return edge_index
