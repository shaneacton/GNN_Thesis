from typing import List, Dict, Any

from torch import Tensor, nn


from Code.Models.GNNs.Layers.base_graph_layer import BaseGraphLayer


class GraphLayer(BaseGraphLayer):

    def __init__(self, sizes: List[int], prep_modules, message_modules,
                 update_modules, activation_type=None, dropout_ratio=None, activation_kwargs=None):

        from Code.Models.GNNs.LayerModules.Message.message_module import MessageModule
        from Code.Models.GNNs.LayerModules.Prepare.prepare_module import PrepareModule
        from Code.Models.GNNs.LayerModules.Update.update_module import UpdateModule

        BaseGraphLayer.__init__(self, sizes, activation_type, dropout_ratio, activation_kwargs)
        self.preparation_modules: List[PrepareModule] = prep_modules
        self.message_modules: List[MessageModule] = message_modules
        self.update_modules: List[UpdateModule] = update_modules

        modules = prep_modules if prep_modules else [] + message_modules if message_modules \
            else [] + update_modules if update_modules else []
        self.all_modules = nn.ModuleList(modules)

    def get_all_arg_names(self):
        """gets all the arguments that this layer may need throughout its prop"""
        all = []
        for prep_module in self.preparation_modules:
            all.extend(self.get_method_arg_names(prep_module.forward))
        for message_module in self.message_modules:
            all.extend(self.get_method_arg_names(message_module.forward))
        for update_module in self.update_modules:
            print("update:",update_module)
            all.extend(self.get_method_arg_names(update_module.forward))
        all += ["edge_index", "edge_types", "batch", "graph", "encoding"]
        return list(set(all))  # returns a list of unique args which are used through this layers propagation

    def prepare(self, data):
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

    def forward(self, data, return_after_prep=False):
        """
        :param return_after_prep: whether to return the result of the preparation module alongside the result of the
            update module
        """
        kwargs = self.prepare(data)
        x_prepped = data.x

        data.x = self.propagate(data.edge_index, x=x_prepped, kwargs=kwargs)

        if return_after_prep:
            return data, x_prepped
        else:
            return data

