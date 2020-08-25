import inspect
from typing import List, Type

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops

from Code.Data.Graph.Embedders.graph_encoding import GraphEncoding


class BaseGraphLayer(MessagePassing):
    """wrapper around a propagation layer such as SAGEConv, GATConv etc"""

    def __init__(self, sizes: List[int]):
        super().__init__()
        self.sizes = sizes

    @property
    def input_size(self):
        return self.sizes[0]

    @property
    def output_size(self):
        return self.sizes[-1]

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def get_method_arg_names(method):
        return inspect.getfullargspec(method)[0]

    @staticmethod
    def get_needed_args(accepted_args, available_args):
        """returns all of the available args which are accepted"""
        # print("getting needed args:",accepted_args, "from:",available_args)
        return {arg: available_args[arg] for arg in available_args.keys() if arg in accepted_args}

    def get_all_arg_names(self):
        raise NotImplementedError()

    def get_required_kwargs_from_batch(self, data: GraphEncoding):
        """
        return only the params of the data object which are needed by the given layer
        """
        layer_args = self.get_all_arg_names()  # needed
        # print("searching for args:",layer_args, "in",data.__dict__)
        data_args = data.__dict__  # given
        present_args = {arg: data_args[arg] for arg in layer_args if arg in data_args.keys()}
        return present_args

    def prepare_args(self, data: GraphEncoding):
        self.clean_loops(data)

        kwargs = self.get_required_kwargs_from_batch(data)
        kwargs["edge_types"] = data.edge_types
        kwargs["node_types"] = data.node_types
        kwargs["encoding"] = data
        kwargs["graph"] = data.graph
        kwargs["layer"] = data.layer

        return kwargs

    @staticmethod
    def clean_loops(data: GraphEncoding):
        # print("cleaning. index:", data.edge_index.size(), "types:", data.edge_types.size())
        try:
            data.edge_index, data.types.edge_types = remove_self_loops(data.edge_index, edge_attr=data.edge_types)
        except Exception as e:
            print("error cleaning loops in edge index:",data.edge_index, "data:", data)
            raise e
        data.edge_index, data.types.edge_types = add_self_loops(data.edge_index, num_nodes=data.x.size(0), edge_weight=data.edge_types)
        # print("after cleaning. index:", data.edge_index.size(), "types:", data.edge_types.size())
