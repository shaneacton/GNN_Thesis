import Code.constants
from Code.Data.Text.Tokenisation.document_extract import DocumentExtract


class Token(DocumentExtract):

    """a token is a token span of length 1"""

    def __init__(self, token_sequence, subtoken_indexes):
        from Code.Config import graph_construction_config
        super().__init__(token_sequence, subtoken_indexes, Code.constants.TOKEN)

    def get_subtype(self):
        return ""