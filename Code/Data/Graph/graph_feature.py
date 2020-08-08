from abc import ABC


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

    def __repr__(self):
        return repr(type(self)) + ": " + self.subtype
