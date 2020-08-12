from typing import Union, Tuple, Dict

import torch

from Code.Data.Graph.graph_feature import GraphFeature


class TypeMap:

    def __init__(self):
        self.type_to_id: Dict[Tuple[type, Union[None, str]], int] = {}
        self.id_to_type: Dict[int, Tuple[type, Union[None, str]]] = {}
        self.next_id = 0

    def register_type(self, type):
        """
        stores newly encountered node/edge types and maps to type id
        this is used to encode node/edge types
        """

        if type in self.type_to_id.keys():
            return
        self.type_to_id[type] = self.next_id
        self.id_to_type[self.next_id] = type
        self.next_id += 1

    def get_id_from_type(self, type):
        self.register_type(type)
        return self.type_to_id[type]

    def get_type_from_id(self, id: Union[int, torch.Tensor]):
        if isinstance(id, torch.Tensor):
            if len(id.size()) > 1 or id.size(0) != 1:
                raise Exception("tensor provided must be single int, got size: " + repr(id.size()))
            id = id.item()
        return self.id_to_type[id]

    def get_id(self, feature: GraphFeature):
        return self.get_id_from_type(feature.get_type())
