from torch import nn
from torch_geometric.data import Batch

from Code.Models.GNNs.Abstract.graph_layer import GraphLayer


class GNN(nn.Module):

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_required_kwargs_from_batch(data: Batch, layer: GraphLayer):
        """
        plucks params out of the data object which are needed by the given layer
        """
        layer_args = layer.get_base_layer_all_arg_names()
        data_args = data.__dict__
        present_args = {arg: data_args[arg] for arg in layer_args if arg in data_args.keys()}
        return present_args
