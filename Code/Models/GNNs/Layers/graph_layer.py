from typing import List, Dict, Any

from torch import Tensor

from Code.Data.Graph.Embedders.graph_encoding import GraphEncoding
from Code.Models.GNNs.LayerModules.Message.attention_module import AttentionModule
from Code.Models.GNNs.LayerModules.Message.message_module import MessageModule
from Code.Models.GNNs.LayerModules.Prepare.prepare_module import PrepareModule
from Code.Models.GNNs.LayerModules.Prepare.relational_prep import RelationalPrep
from Code.Models.GNNs.LayerModules.update_module import UpdateModule
from Code.Models.GNNs.Layers.base_graph_layer import BaseGraphLayer


class GraphLayer(BaseGraphLayer):

    def __init__(self, sizes: List[int]):

        super().__init__(sizes)
        self.preparation_module: PrepareModule = RelationalPrep(self.input_size, self.output_size, 3)
        self.message_module: MessageModule = AttentionModule(self.output_size, heads=8)
        self.update_module: UpdateModule = UpdateModule()

    def get_all_arg_names(self):
        """gets all the arguments that this layer may need throughout its prop"""
        all = self.get_method_arg_names(self.preparation_module.forward)
        all += self.get_method_arg_names(self.message_module.forward)
        all += self.get_method_arg_names(self.update_module.forward)
        all += ["edge_index", "edge_types", "batch", "graph", "encoding"]
        return list(set(all))  # returns a list of unique args which are used through this layers propagation

    def prepare(self, data: GraphEncoding):
        """prepares args and passes input through the preparation module"""
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
        print("layer received:", data.x)
        kwargs = self.prepare(data)
        x = data.x
        print("passing through, ", self, "x:", x.size())
        print("x after prep:", x)

        x = self.propagate(data.edge_index, x=x, kwargs=kwargs)

        data.x = x
        return data


if __name__ == "__main__":
    layer = GraphLayer([1, 2])
    print(layer)
