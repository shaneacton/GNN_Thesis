from typing import List, Type

import torch
from torch import Tensor, tensor
from torch_geometric.data import Batch

from Code.Models.GNNs.graph_layer import GraphLayer
from Code.Training import device


class RelationalLayer(GraphLayer):

    """graph layer which does not propagate info along edges unless that edge is of a target type"""

    def __init__(self, sizes: List[int], layer_type: Type[GraphLayer], target_edge_type, activation_type=None, layer_args=None):
        self.target_edge_type = target_edge_type
        super().__init__(sizes, layer_type, activation_type, layer_args)

    def message(self, x_j, **kwargs):
        """overrides the base graph layers message passing to return only zeros when edge type is not target"""
        # x_j has shape [num_edges, num_features]
        # not all edges should send a message, some should be zero'd out

        edge_types = kwargs["edge_types"]
        edge_mask = self.get_edge_mask(edge_types)
        x_j = edge_mask * x_j  # blanks out node messages on edges which are not target type
        return super(RelationalLayer, self).message(x_j, **kwargs)

    def get_edge_mask(self, edge_types: Tensor):
        mask = tensor([1 if edge_types[i]==self.target_edge_type else 0 for i in range(edge_types.size(0))],
                      dtype=torch.long)
        # mask must be of shape [num_edges, 1]
        return mask.view(-1, 1).to(device)



