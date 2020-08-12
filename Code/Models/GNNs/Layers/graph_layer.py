import inspect
from typing import List, Type, Dict, Any

from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops

from Code.Data.Graph.Embedders.graph_encoding import GraphEncoding


class GraphLayer(MessagePassing):
    """wrapper around a propagation layer such as SAGEConv, GATConv etc"""

    def __init__(self, sizes: List[int], layer_type: Type[MessagePassing], activation_type=None, layer_args=None, init_layer=True):
        super().__init__()
        self.layer_type = layer_type
        self.sizes = sizes
        self.layer_args = layer_args if layer_args else {}

        if init_layer:
            self.layer: MessagePassing = self.initialise_layer()
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

    def get_method_arg_names(self, method):
        return inspect.getfullargspec(method)[0]

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
        # print("getting needed args:",accepted_args, "from:",available_args)
        return {arg: available_args[arg] for arg in available_args.keys() if arg in accepted_args}

    def message(self, x_i: Tensor, x_j: Tensor, size_i: Tensor, size_j: Tensor, edge_index_i: Tensor,
                edge_index_j: Tensor, kwargs: Dict[str, Any]):

        #  the following variables have just been created by pytorch geometric - inject them into our kwargs obj
        kwargs["x_i"] = x_i
        kwargs["x_j"] = x_j
        kwargs["size_i"] = size_i
        kwargs["size_j"] = size_j
        kwargs["edge_index_i"] = edge_index_i
        kwargs["edge_index_j"] = edge_index_j

        needed_args = self.get_needed_args(self.get_base_layer_message_arg_names(), kwargs)
        return self.layer.message(**needed_args)

    def update(self, inputs, kwargs: Dict[str, Any]):
        needed_args = self.get_needed_args(self.get_base_layer_update_arg_names(), kwargs)
        return self.layer.update(inputs, **needed_args)

    def get_required_kwargs_from_batch(self, data: GraphEncoding):
        """
        return only the params of the data object which are needed by the given layer
        """
        layer_args = self.get_base_layer_all_arg_names()  # needed
        # print("searching for args:",layer_args, "in",data.__dict__)
        data_args = data.__dict__  # given
        present_args = {arg: data_args[arg] for arg in layer_args if arg in data_args.keys()}
        return present_args

    def prepare_args(self, data: GraphEncoding):
        self.clean_loops(data)

        kwargs = self.get_required_kwargs_from_batch(data)
        kwargs = self.get_kwargs_with_defaults(kwargs)
        kwargs["edge_types"] = data.edge_types
        kwargs["node_types"] = data.node_types

        return kwargs

    def forward(self, data: GraphEncoding):
        kwargs = self.prepare_args(data)

        # print("propping args:", kwargs)
        x = self.propagate(data.edge_index, x=data.x, kwargs=kwargs)
        if self.activation:
            x = self.activation(x)
        data.x = x
        return data

    @staticmethod
    def clean_loops(data: GraphEncoding):
        print("cleaning. index:", data.edge_index.size(), "types:", data.edge_types.size())
        try:
            data.edge_index, data.types.edge_types = remove_self_loops(data.edge_index, edge_attr=data.edge_types)
        except Exception as e:
            print("error cleaning loops in edge index:",data.edge_index, "data:", data)
            raise e
        data.edge_index, data.types.edge_types = add_self_loops(data.edge_index, num_nodes=data.x.size(0), edge_weight=data.edge_types)
        print("after cleaning. index:", data.edge_index.size(), "types:", data.edge_types.size())
