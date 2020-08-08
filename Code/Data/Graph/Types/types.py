import torch

from Code.Data.Graph import TypeMap


class Types:

    def __init__(self, type_map: TypeMap, node_types: torch.Tensor, edge_types: torch.Tensor):
        self.type_map: TypeMap = type_map
        self.node_types: torch.Tensor = node_types
        self.edge_types: torch.Tensor = edge_types

