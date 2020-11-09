from typing import List

import torch

from Code.Data.Graph.Types.type_map import TypeMap


class Types:

    """stores the id<->type maps for nodes and edges, as well as the type vectors"""

    def __init__(self, node_type_map: TypeMap, edge_type_map: TypeMap, node_types: torch.Tensor, edge_types: torch.Tensor):
        """
        :param node_types: shape (N,1)
        :param edge_types: shape (E, 1)
        """
        self.node_type_map = node_type_map
        self.edge_type_map = edge_type_map
        self.node_types: torch.Tensor = node_types
        self.edge_types: torch.Tensor = edge_types

    @staticmethod
    def from_types_list(types: List):
        types: List[Types] = types
        n_map = TypeMap.from_typemap_list([t.node_type_map for t in types])
        e_map = TypeMap.from_typemap_list([t.edge_type_map for t in types])

        n_types = []
        e_types = []
        for t in range(len(types)):
            ty = types[t]
            num_nodes = ty.node_types.size(0)
            n_types.append(ty.node_types + num_nodes)
            e_types.append(ty.edge_types + num_nodes)

        n_types = torch.cat(n_types)
        e_types = torch.cat(e_types)

        return Types(n_map, e_map, n_types, e_types)