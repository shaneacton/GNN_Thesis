from Code.Data.Graph.Edges.edge_relation import EdgeRelation


class DocumentEdge(EdgeRelation):

    """
    these links represent document structure
    types include:
    'codoc' - co-occurence in the same document
    'costat' - co-occurence in the same statement
    'stat2word' - statement-level-node to  entity nodes in that statement
    'doc2stat' - document-level-node to statement-level-node for all statements in the doc
    """

    def get_label(self):
        return "DOC"

    def __init__(self, from_id, to_id, subtype: str):
        super().__init__(from_id, to_id, directed='2' in subtype)
        self.subtype = subtype