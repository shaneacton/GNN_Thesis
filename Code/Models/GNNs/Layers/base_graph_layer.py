from typing import List

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops

from Code.Data.Graph.graph_encoding import GraphEncoding
from Code.Models.GNNs.gnn_component import GNNComponent


class BaseGraphLayer(GNNComponent, MessagePassing):
    """wrapper around a propagation layer such as SAGEConv, GATConv etc"""

    def __init__(self, sizes: List[int], activation_type=None, dropout_ratio=None, activation_kwargs=None):
        MessagePassing.__init__(self)
        GNNComponent.__init__(self, sizes, activation_type, dropout_ratio, activation_kwargs=activation_kwargs)

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
        kwargs["types"] = data.edge_types
        kwargs["node_types"] = data.node_types
        kwargs["encoding"] = data
        kwargs["graph"] = data.graphs
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
