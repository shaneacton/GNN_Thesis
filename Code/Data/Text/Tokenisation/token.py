from Code.Config import configuration
from Code.Data.Text.Tokenisation.document_extract import DocumentExtract


class Token(DocumentExtract):

    def __init__(self, token_sequence, subtoken_indexes):
        super().__init__(token_sequence, subtoken_indexes, configuration.TOKEN)

    def get_subtype(self):
        return ""