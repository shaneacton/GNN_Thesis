from Code.Data.Graph.Edges.edge_relation import EdgeRelation


class QueryEdge(EdgeRelation):

    def __init__(self, from_id, to_id, query_level, context_level):
        subtype = query_level + "2" + context_level
        super().__init__(from_id, to_id, subtype=subtype)
        self.query_level = query_level
        self.context_level = context_level

    def get_label(self):
        return self.subtype

