import inspect
from typing import List, Dict, Any

from torch import Tensor

from Code.Data.Graph.Embedders.graph_encoding import GraphEncoding
from Code.Models.GNNs.LayerModules.attention_module import AttentionModule
from Code.Models.GNNs.LayerModules.message_module import MessageModule
from Code.Models.GNNs.LayerModules.prepare_module import PrepareModule
from Code.Models.GNNs.LayerModules.relational_prep import RelationalPrep
from Code.Models.GNNs.LayerModules.update_module import UpdateModule
from Code.Models.GNNs.Layers.graph_layer import GraphLayer


class NewLayer(GraphLayer):

    def __init__(self, sizes: List[int]):

        super().__init__(sizes, None, init_layer=False)
        self.preparation_module: PrepareModule = RelationalPrep(self.input_size, self.output_size, 3)
        self.message_module: MessageModule = AttentionModule(self.input_size, self.output_size)
        self.update_module: UpdateModule = UpdateModule()

    def get_base_layer_all_arg_names(self):
        """gets all the arguments that this layer would need throughout its prop"""
        all = self.get_method_arg_names(self.preparation_module.forward)
        all += self.get_method_arg_names(self.message_module.forward)
        all += self.get_method_arg_names(self.update_module.forward)
        all += ["edge_index", "edge_types", "batch"]
        return list(set(all))  # returns a list of unique args which are used through this layers propagation

    def get_base_layer_forward_defaults(self):
        """
        gets the defaulted values from the forward method
        these defaulted args often become required when moving from forward to message
        thus they must be passed explicitly into message if needed
        """
        return {}
        all_args = self.get_base_layer_forward_arg_names()
        default_values = inspect.getfullargspec(self.get_layer().forward)[3]
        if default_values is None:
            return {}

        defaulted_args = all_args[len(all_args) - len(default_values):]
        return {defaulted_args[i]: default_values[i] for i in range(len(defaulted_args))}

    def prepare(self, data: GraphEncoding):
        kwargs = self.prepare_args(data)
        needed_args = self.get_needed_args(self.get_method_arg_names(self.preparation_module.forward), kwargs)
        x = self.preparation_module(**needed_args)
        data.x = x
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

        needed_args = self.get_needed_args(self.get_method_arg_names(self.message_module.forward), kwargs)
        return self.message_module(**needed_args)

    def update(self, inputs, kwargs: Dict[str, Any]):
        needed_args = self.get_needed_args(self.get_method_arg_names(self.update_module.forward), kwargs)
        return self.update_module(inputs, **needed_args)

    def forward(self, data: GraphEncoding):
        kwargs = self.prepare(data)
        x = data.x
        print("x:", x.size())

        x = self.propagate(data.edge_index, x=x, kwargs=kwargs)
        if self.activation:
            x = self.activation(x)

        data.x = x
        return data


if __name__ == "__main__":
    layer = NewLayer([1,2])
    print(layer)
