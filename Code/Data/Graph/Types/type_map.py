from typing import Union, Tuple, Dict, List

import torch

from Code.Data.Graph.graph_feature import GraphFeature


class TypeMap:

    """
        a bidirectional mapping between node/edge types and type ids
        a type is defined as a (class_type, subtype-string) tuple
    """

    def __init__(self, batch_offset=0):
        self._type_to_id: Dict[Tuple[type, Union[None, str]], int] = {}
        self._id_to_type: Dict[int, Tuple[type, Union[None, str]]] = {}
        self._next_id = 0

    @staticmethod
    def from_typemap_list(maps: List):
        """
            type_maps from similar data populations, with the same graph construction processes will likely be the same
            however we add a safety measure to ensure no types are missed during batching
        """
        map = maps[0]
        for m in range(1, len(maps)):
            map += maps[m]
        return map

    def __add__(self, other):
        other: TypeMap = other
        for type in other._type_to_id:
            self.register_type(type)
        return self

    def register_type(self, type):
        """
        stores newly encountered node/edge types and maps to type id
        this is used to encode node/edge types
        """

        if type in self._type_to_id.keys():
            return
        self._type_to_id[type] = self._next_id
        self._id_to_type[self._next_id] = type
        self._next_id += 1

    def get_id_from_type(self, type):
        self.register_type(type)
        id = self._type_to_id[type]
        return id

    def get_type_from_id(self, id: Union[int, torch.Tensor]):
        if isinstance(id, torch.Tensor):
            if len(id.size()) > 1 or id.size(0) != 1:
                raise Exception("tensor provided must be single int, got size: " + repr(id.size()))
            id = id.item()
        return self._id_to_type[id]

    def get_id(self, feature: GraphFeature):
        return self.get_id_from_type(feature.get_type())
