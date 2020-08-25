from Code.Data.Graph.Edges.edge_relation import EdgeRelation
from Code.Config import graph_construction_config as construction


class CandidateEdge(EdgeRelation):
    def __init__(self, from_id, to_id, context_level, connection_source):
        subtype = construction.CANDIDATE + "2" + connection_source + ":" + context_level
        super().__init__(from_id, to_id, subtype=subtype)
        self.context_level = context_level

