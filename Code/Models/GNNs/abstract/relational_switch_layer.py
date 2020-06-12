from typing import List, Dict, Type, Any, Union

import torch
from torch import nn, Tensor
from torch_geometric.nn import SAGEConv, MessagePassing

from Code.Data.Graph import example_batch
from Code.Models.GNNs.abstract.graph_layer import GraphLayer
from Code.Models.GNNs.abstract.relational_layer import RelationalLayer
from Code.Training import device


class RelationalSwitchLayer(GraphLayer):
    """
    a layer which switches the sub_layer used based on edge type id
    each sublayer is wrapped in a relational layer which ensures only messages passed along target edge types are sent
    overrides the message function to sum the messages of each edge type sub_layer

    the update/aggregation functions need only be learned once. Thus a random sub_layer will be chosen as the
    representative of this switch layer, and it will be used for updates/ aggregations
    """

    EDGE_TYPES = "edge_types"

    def __init__(self, sizes: List[int], layer_type: Type[Union[GraphLayer, MessagePassing]], sub_layer_args=None):
        self.type_to_layer_map: Dict[str, GraphLayer] = {}
        self.sub_layer_args = sub_layer_args
        self.sub_layer_type = layer_type
        super().__init__(sizes, RelationalSwitchLayer)
        self.sub_layers: Dict[Any, RelationalLayer] = None
        self.representative: RelationalLayer = None  # used for the update function

    def initialise_layer(self):
        return None

    def get_layer(self):
        """
        returns a sample sublayer ie sub[0]
        this method is primarily used for argument resolution and each sub_layer has the same args
        """
        return self.representative.get_layer()

    def forward(self, x, edge_index, batch, **kwargs):
        if self.sub_layers is None:
            edge_types = kwargs["edge_types"]
            self.create_sub_layers(edge_types)

        return super(RelationalSwitchLayer, self).forward(x, edge_index, batch, **kwargs)

    def message(self, x_j, kwargs: Dict[str, Any]):
        """
        groups and sums the messages sent by each different edge type. then sends summed message off to wrapped layer
        """

        num_edges = x_j.size(0)
        edge_messages = [edge_layer.message(x_j, kwargs=kwargs) for edge_layer in self.sub_layers.values()]
        # print("edge messages:",edge_messages)
        edge_messages = torch.stack(edge_messages, dim=0)
        edge_messages = torch.sum(edge_messages, dim=0).view(num_edges, -1)
        # print("summed_edge_messages:",edge_messages, edge_messages.size())
        return edge_messages

    def update(self, inputs, kwargs: Dict[str, Any]):
        return self.representative.update(inputs, kwargs=kwargs)

    def create_sub_layers(self, edge_types: Tensor):
        """
        creates a sublayer for each unique type found in edge types
        """
        # edge types is [num_nodes, 1]
        unique_types = set([RelationalLayer.get_clean_type(edge_types[i]) for i in range(edge_types.size(0))])
        for type in unique_types:
            self.create_layer_for_type(type)
        for sub in self.type_to_layer_map.values():
            self.representative = sub  # assigns an arbitrary sub_layer to be the representative
            break

        print("created layers:", self.type_to_layer_map)
        self.sub_layers = nn.ModuleDict(self.type_to_layer_map)

    def create_layer_for_type(self, type_id):
        """
        each created layer is a relational_layer meaning it will only send messages along edges of the correct type
        """
        if not isinstance(type_id, str):
            type_id = repr(type_id)

        if type_id in self.type_to_layer_map.keys():
            return self.type_to_layer_map[type_id]
        # first pass with this newly encountered edge type, must create new sub_layer

        new_layer = RelationalLayer(self.sizes, self.sub_layer_type, type_id, layer_args=self.sub_layer_args).to(device)
        self.type_to_layer_map[type_id] = new_layer


if __name__ == "__main__":
    rel_switch = RelationalSwitchLayer([3, 128], SAGEConv, sub_layer_args={}).to(device)
    print(rel_switch)
    print("before forward num params:", rel_switch.num_params)

    out = rel_switch(example_batch.x, example_batch.edge_index, example_batch.batch, edge_types=example_batch.edge_types)
    print(rel_switch)
    print("after forward num params:", rel_switch.num_params)