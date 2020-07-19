from Code.Data.Graph.Edges.edge_relation import EdgeRelation


class WindowEdge(EdgeRelation):

    def __init__(self, from_id, to_id, subtype: str, level_id):
        super().__init__(from_id, to_id, subtype=subtype)
        self.level_id = level_id

    def get_label(self):
        return self.subtype.upper()


