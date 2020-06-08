from typing import List, Dict

from torch import nn

from Code.Models.GNNs.graph_layer import GraphLayer


class RelationalSwitchLayer(GraphLayer):
    """
        a layer which switches the sub_layer used based on the given layer type id
        """

    EDGE_TYPE = "edge_type"

    def __init__(self, sizes: List[int], layer_type: type, sub_layer_args=None):
        self.type_to_layer_map: Dict[str, GraphLayer] = {}
        self.sub_layer_args = sub_layer_args
        self.sub_layers = None
        self.sub_layer_type = layer_type
        super().__init__(sizes, RelationalSwitchLayer)

    def forward(self, x, edge_index, edge_types, batch, **kwargs):
        type_id = kwargs[RelationalSwitchLayer.EDGE_TYPE]
        layer = self.get_layer(type_id)
        return layer(x, edge_index, edge_types, batch, **kwargs)

    def get_layer(self, type_id):
        if isinstance(type_id, int):
            type_id = repr(type_id)

        if type_id in self.type_to_layer_map.keys():
            return self.type_to_layer_map[type_id]
        # first pass with this newly encountered edge type, must create new sub_layer

        new_layer = self.layer_type(self.sizes, **self.sub_layer_args)
        self.type_to_layer_map[type_id] = new_layer
        self.sub_layers = nn.ModuleDict(self.type_to_layer_map)
        return new_layer