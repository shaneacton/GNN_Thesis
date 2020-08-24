from typing import List, Dict

from Code.Config import gnn_config, GNNConfig
from Code.Models.GNNs.Layers.base_graph_layer import BaseGraphLayer
from Code.Models.GNNs.Layers.graph_layer import GraphLayer
from Code.Training import device


class LayerConstructor:
    """creates a gnn layer from a gnn layer config entry"""

    def get_layer(self, sizes: List[int], args:Dict):
        prep_modules = self.get_layer_modules(gnn_config.PREPARATION_MODULES, sizes, args)
        message_modules = self.get_layer_modules(gnn_config.MESSAGE_MODULES, sizes, args)
        update_modules = self.get_layer_modules(gnn_config.UPDATE_MODULES, sizes, args)

        layer = GraphLayer(sizes, prep_modules, message_modules, update_modules)
        return layer.to(device)

    def get_layer_modules(self, module_group, sizes, args):
        modules = []

        all_module_args = args[module_group]
        for module_args in all_module_args:
            module_type = module_args[gnn_config.MODULE_TYPE]
            module_args["in_channels"] = sizes[0]
            module_args["out_channels"] = sizes[1]
            module_args["channels"] = sizes[1]

            needed_args = BaseGraphLayer.get_method_arg_names(module_type.__init__)
            module_args = BaseGraphLayer.get_needed_args(needed_args, module_args)

            module = module_type(**module_args).to(device)
            modules.append(module)

        return modules


if __name__ == "__main__":
    const = LayerConstructor()
    conf = GNNConfig()
    args = conf.layers[0]
    const.get_layer([5,8], args)
