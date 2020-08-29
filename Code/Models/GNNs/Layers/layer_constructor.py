import copy
from typing import List, Dict

from Code.Config import gnn_config, GNNConfig
from Code.Models.GNNs.Layers.base_graph_layer import BaseGraphLayer
from Code.Models.GNNs.Layers.graph_layer import GraphLayer
from Code.Training import device


class LayerConstructor:
    """creates a gnn layer from a gnn layer config entry"""

    def get_layer(self, sizes: List[int], layer_args: Dict, global_args: Dict):
        prep_modules = self.get_layer_modules(gnn_config.PREPARATION_MODULES, sizes, layer_args, global_args)
        message_modules = self.get_layer_modules(gnn_config.MESSAGE_MODULES, sizes, layer_args, global_args)
        update_modules = self.get_layer_modules(gnn_config.UPDATE_MODULES, sizes, layer_args, global_args)

        layer = GraphLayer(sizes, prep_modules, message_modules, update_modules)
        return layer.to(device)

    def get_layer_modules(self, module_group, sizes, layer_args, global_args):
        """

        :param module_group: Prep/ Message/ Update
        :param layer_args: list of args for each of the (Prep/ Message/ Update) modules
        """
        modules = []

        all_module_args = layer_args[module_group]
        for module_args in all_module_args:
            args = copy.deepcopy(global_args)
            # print("global args:", args)
            args.update(module_args)
            # print("effective args:",args)

            module_type = args[gnn_config.MODULE_TYPE]
            args["in_channels"] = sizes[0]
            args["out_channels"] = sizes[1]
            args["channels"] = sizes[1]

            needed_args = BaseGraphLayer.get_method_arg_names(module_type.__init__)
            args = BaseGraphLayer.get_needed_args(needed_args, args)
            # print("needed:", needed_args, "using:", args)

            module = module_type(**args).to(device)
            modules.append(module)

        return modules


if __name__ == "__main__":
    const = LayerConstructor()
    conf = GNNConfig()
    args = conf.layers[0]
    const.get_layer([5,8], args)
