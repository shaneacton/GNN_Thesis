from typing import List, Dict, Any, Type

from torch import Tensor, nn

from Code.Data.Graph.Embedders.graph_encoding import GraphEncoding
from Code.Models.GNNs.LayerModules.Message.message_module import MessageModule
from Code.Models.GNNs.LayerModules.Prepare.prepare_module import PrepareModule
from Code.Models.GNNs.LayerModules.update_module import UpdateModule
from Code.Models.GNNs.Layers.base_graph_layer import BaseGraphLayer


class GraphLayer(BaseGraphLayer):

    def __init__(self, sizes: List[int], prep_modules: List[PrepareModule], message_modules: List[MessageModule],
                 update_modules: List[UpdateModule]):

        super().__init__(sizes)
        self.preparation_modules: List[PrepareModule] = prep_modules
        self.message_modules: List[MessageModule] = message_modules
        self.update_modules: List[UpdateModule] = update_modules

        self.all_modules = nn.ModuleList(prep_modules + message_modules + update_modules)

    def get_all_arg_names(self):
        """gets all the arguments that this layer may need throughout its prop"""
        all = []
        for prep_module in self.preparation_modules:
            all.extend(self.get_method_arg_names(prep_module.forward))
        for message_module in self.message_modules:
            all.extend(self.get_method_arg_names(message_module.forward))
        for update_module in self.update_modules:
            all.extend(self.get_method_arg_names(update_module.forward))
        all += ["edge_index", "edge_types", "batch", "graph", "encoding"]
        return list(set(all))  # returns a list of unique args which are used through this layers propagation

    def prepare(self, data: GraphEncoding):
        """prepares args and passes input through the preparation module"""
        kwargs = self.prepare_args(data)
        for prep_module in self.preparation_modules:
            needed_args = self.get_needed_args(self.get_method_arg_names(prep_module.forward), kwargs)
            kwargs["x"] = prep_module(**needed_args)

        data.x = kwargs["x"]
        return kwargs

    def message(self, x_i: Tensor, x_j: Tensor, size_i: Tensor, size_j: Tensor, edge_index_i: Tensor,
                edge_index_j: Tensor, kwargs: Dict[str, Any]):

        #  the following variables have just been created by pytorch geometric - inject them into our kwargs obj
        kwargs["x_i"] = x_i
        kwargs["x_j"] = x_j
        kwargs["size_i"] = size_i
        kwargs["size_j"] = size_j
        kwargs["edge_index_i"] = edge_index_i
        kwargs["edge_index_j"] = edge_index_j

        for message_module in self.message_modules:
            needed_args = self.get_needed_args(self.get_method_arg_names(message_module.forward), kwargs)
            kwargs["x_j"] = message_module(**needed_args)
        return kwargs["x_j"]

    def update(self, aggr_out, kwargs: Dict[str, Any]):
        for update_module in self.update_modules:
            needed_args = self.get_needed_args(self.get_method_arg_names(update_module.forward), kwargs)
            kwargs["x"] = update_module(aggr_out, **needed_args)
        return kwargs["x"]

    def forward(self, data: GraphEncoding):
        kwargs = self.prepare(data)
        x = data.x

        x = self.propagate(data.edge_index, x=x, kwargs=kwargs)

        data.x = x
        return data


if __name__ == "__main__":
    layer = GraphLayer([1, 2])
    print(layer)
