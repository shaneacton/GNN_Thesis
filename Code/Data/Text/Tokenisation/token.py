from Code.Config import graph_construction_config
from Code.Data.Text.Tokenisation.document_extract import DocumentExtract


class Token(DocumentExtract):

    def __init__(self, token_sequence, subtoken_indexes):
        super().__init__(token_sequence, subtoken_indexes, graph_construction_config.TOKEN)

    def get_subtype(self):
        return ""