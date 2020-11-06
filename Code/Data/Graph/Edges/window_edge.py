from Code.Data.Graph.Edges.edge_relation import EdgeRelation


class WindowEdge(EdgeRelation):

    def __init__(self, from_id, to_id, subtype: str, distance):
        super().__init__(from_id, to_id, subtype=subtype, directed=False)
        self.distance = distance

    def get_label(self):
        return self.subtype.upper()


