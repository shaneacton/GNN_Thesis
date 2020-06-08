from Code.Data.Graph.Edges.edge_relation import EdgeRelation
from Code.Data.Text.Tokenisation.document_extract import DocumentExtract


class DocumentEdge(EdgeRelation):

    """
    these links represent document structure
    types include:
    'codoc' - co-occurence in the same document
    'costat' - co-occurence in the same statement

    'stat2word','pass2stat', 'pass2word', 'doc2pass', 'doc2stat', 'doc2word'
    all the x2y types denote doc(ument)/stat(ement)/pass(age)/word  level node connections
    """

    DODOC = "codoc"
    COSTAT = "costat"

    def get_label(self):
        return self.subtype

    def __init__(self, from_id, to_id, subtype: str):
        super().__init__(from_id, to_id, directed='2' in subtype)
        self.set_subtype(subtype)

    @staticmethod
    def get_x2y_edge_type(from_level:str, to_level:str):
        return from_level + "2" + to_level
