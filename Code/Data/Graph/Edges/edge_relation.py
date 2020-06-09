from abc import ABC, abstractmethod

from Code.Data.Graph.graph_feature import GraphFeature


class EdgeRelation(GraphFeature, ABC):

    def __init__(self, from_id, to_id, directed=True, subtype=None, direction=None):
        """
        :param direction: should be made "reverse" if edge is directed and this instance is the returning edge
        """
        if direction is None:
            direction = "uni" if directed else "forward"
        self.direction = direction
        super().__init__(subtype=subtype)
        self.from_id = from_id
        self.to_id = to_id
        self.directed = directed

    def __getitem__(self, item):
        if item == 0:
            return self.from_id
        if item == 1:
            return self.to_id
        raise Exception()

    def __eq__(self, other):
        type_eq = type(self) == type(other) and self.subtype == other.subtype
        top_eq = (self.from_id == other.from_id and self.to_id == other.to_id)
        top_eq = top_eq or ((not self.directed) and self.from_id == other.to_id and self.to_id == other.from_id)
        return top_eq and type_eq

    def __hash__(self):
        ids = [self.from_id, self.to_id]
        ids = ids if self.directed else sorted(ids)  # if sorted, each dir will has the same
        return hash((tuple(ids), type(self)))

    def get_type(self):
        return super(EdgeRelation, self).get_type() + (self.direction,)

    @abstractmethod
    def get_label(self):
        raise NotImplementedError()