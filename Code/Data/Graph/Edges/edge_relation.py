from abc import ABC, abstractmethod


class EdgeRelation(ABC):

    def __init__(self, from_id, to_id, directed=True):
        self.from_id = from_id
        self.to_id = to_id
        self.directed = directed
        self.subtype = None

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
        return type(self), self.subtype

    @abstractmethod
    def get_label(self):
        raise NotImplementedError()
