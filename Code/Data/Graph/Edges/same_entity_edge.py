from Code.Data.Graph.Edges.edge_relation import EdgeRelation


class SameEntityEdge(EdgeRelation):

    def __init__(self, from_id, to_id, is_coref):
        super().__init__(from_id, to_id, directed=is_coref)
        self.is_coref = is_coref

    def get_label(self):
        return "COREF" if self.is_coref else "SAME"