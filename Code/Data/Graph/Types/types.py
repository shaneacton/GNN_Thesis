import torch

from Code.Data.Graph.Types.type_map import TypeMap


class Types:

    def __init__(self, node_type_map: TypeMap, edge_type_map: TypeMap, node_types: torch.Tensor, edge_types: torch.Tensor):
        self.node_type_map = node_type_map
        self.edge_type_map = edge_type_map
        self.node_types: torch.Tensor = node_types
        self.edge_types: torch.Tensor = edge_types

