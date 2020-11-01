import Code.constants
from Code.Data.Graph.Edges.edge_relation import EdgeRelation


class CandidateEdge(EdgeRelation):
    def __init__(self, from_id, to_id, context_level, connection_source):
        subtype = Code.constants.CANDIDATE + "2" + connection_source + ":" + context_level
        super().__init__(from_id, to_id, subtype=subtype)
        self.context_level = context_level


