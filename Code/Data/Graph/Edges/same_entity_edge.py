from Code.Data.Graph.Edges.edge_relation import EdgeRelation

COMENTION = "comention"
COREFERENCE = "coreference"
UNIQUE_REFERENCE = "unique_ref"


class SameEntityEdge(EdgeRelation):

    """
    connects word nodes which logically represent the same entity
    this can be from mention -> mention : undirected
    or mention -> coref : directed
    or mention -> unique ent : directed
    or coref -> unique ent
    """

    def __init__(self, from_id, to_id, to_coref, to_unique_node=False):
        directed = to_coref or to_unique_node
        self.is_coref = to_coref
        self.is_unique = to_unique_node
        super().__init__(from_id, to_id, directed=directed, subtype=self.get_subtype())

    def get_label(self):
        return self.get_subtype()

    def get_subtype(self):
        if not self.is_coref and not self.is_unique:
            return COMENTION
        if self.is_coref and self.is_unique:
            return COREFERENCE + "2" + UNIQUE_REFERENCE
        if self.is_coref:
            return COREFERENCE
        if self.is_unique:
            return UNIQUE_REFERENCE
