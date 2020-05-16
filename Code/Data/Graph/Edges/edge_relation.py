
class EdgeRelation:

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
        top_eq = top_eq or (self.directed and self.from_id == other.to_id and self.to_id == other.from_id)
        return top_eq and type_eq
