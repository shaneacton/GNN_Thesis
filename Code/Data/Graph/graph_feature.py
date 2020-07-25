from abc import ABC

import torch

from Code.Data.Graph import type_map
from Code.Training import device


class GraphFeature(ABC):

    """the abstract form of a graph node or edge"""

    def __init__(self, subtype=None):
        self.subtype = subtype

    def set_subtype(self, subtype):
        self.subtype = subtype

    def get_type(self):
        """
        returns type, subtype
        only some nodes/edges have subtypes
        """
        return type(self), self.subtype

    def get_type_tensor(self):
        type_id = type_map.get_id_from_type(self.get_type())
        return torch.tensor([type_id]).to(device)

    def __repr__(self):
        return repr(type(self)) + ": " + self.subtype
