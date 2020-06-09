from typing import List, Dict, Type, Any

import torch
from torch import nn, Tensor

from Code.Models.GNNs.graph_layer import GraphLayer
from Code.Models.GNNs.relational_layer import RelationalLayer


class RelationalSwitchLayer(GraphLayer):
    """
    a layer which switches the sub_layer used based on the given edge type id
    each sublayer is wrapped in a relational layer which ensures only messages passed along target edge types are sent
    should override the message function to sum the messages of each edge type sub_layer
    """

    EDGE_TYPES = "edge_types"

    def __init__(self, sizes: List[int], layer_type: Type[GraphLayer], sub_layer_args=None):
        self.type_to_layer_map: Dict[str, GraphLayer] = {}
        self.sub_layer_args = sub_layer_args
        self.sub_layers: Dict[Any, RelationalLayer] = None
        self.sub_layer_type = layer_type
        super().__init__(sizes, RelationalSwitchLayer)

    def forward(self, x, edge_index, edge_types, batch, **kwargs):
        if self.sub_layers is None:
            self.create_sub_layers(edge_types)

        return self.propagate(edge_index, x=x, edge_types=edge_types, batch=batch, **kwargs)

    def message(self, x_j, **kwargs):
        """
        groups and sums the messages sent by each different edge type. then sends summed message off to wrapped layer
        """

        num_nodes = x_j.size(0)
        edge_messages = torch.stack([edge_layer.message(x_j, **kwargs) for edge_layer in self.sub_layers.values()], dim=0)
        edge_messages = torch.sum(edge_messages, dim=0).view(num_nodes,-1)
        return super(RelationalSwitchLayer, self).message(edge_messages, **kwargs)

    def create_sub_layers(self, edge_types: Tensor):
        """
        creates a sublayer for each unique type found in edge types
        """
        # edge types is [num_nodes, 1]
        unique_types = set([edge_types[i] for i in range(edge_types.size(0))])
        for type in unique_types:
            self.create_layer_for_type(type)

        self.sub_layers = nn.ModuleDict(self.type_to_layer_map)

    def create_layer_for_type(self, type_id):
        """
        each created layer is a relational_layer meaning it will only send messages along edges of the correct type
        """
        if isinstance(type_id, int):
            type_id = repr(type_id)

        if type_id in self.type_to_layer_map.keys():
            return self.type_to_layer_map[type_id]
        # first pass with this newly encountered edge type, must create new sub_layer

        new_layer = RelationalLayer(self.sizes, self.sub_layer_type, type_id, layer_args=self.sub_layer_args)
        new_layer = self.layer_type(self.sizes, **self.sub_layer_args)
        self.type_to_layer_map[type_id] = new_layer

if __name__ == "__main__":
    from torch_geometric.data import Data, Batch

    x = torch.tensor([[2, 1, 3], [5, 6, 4], [3, 7, 5], [12, 0, 6]], dtype=torch.float)
    y = torch.tensor([0, 1, 0, 1], dtype=torch.float)

    edge_index = torch.tensor([[0, 2, 1, 0, 3],
                               [3, 1, 0, 1, 2]], dtype=torch.long)

    edge_types = torch.tensor([[0, 2, 1, 0, 3]], dtype=torch.long)

    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_types)

    batch = Batch.from_data_list([data, data])