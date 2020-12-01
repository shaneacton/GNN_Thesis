import Code.constants
from Code.Data.Graph.Edges.edge_relation import EdgeRelation


class CandidateEdge(EdgeRelation):
    def __init__(self, from_id, to_id, connection_level, connection_source):
        subtype = Code.constants.CANDIDATE + "2" + connection_source + ":" + connection_level
        super().__init__(from_id, to_id, subtype=subtype)
        self.connection_level = connection_level


