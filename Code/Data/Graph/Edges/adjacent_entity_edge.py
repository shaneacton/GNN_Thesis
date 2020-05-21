from Code.Data.Graph.Edges.edge_relation import EdgeRelation


class AdjacentEntityEdge(EdgeRelation):

    def __init__(self, from_id, to_id):
        super().__init__(from_id, to_id)

    def get_label(self):
        return "SEQ"


