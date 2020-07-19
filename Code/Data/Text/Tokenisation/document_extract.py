from spacy.tokens.span import Span

from Code.Data.Text.Tokenisation.token_span import TokenSpan


class DocumentExtract(TokenSpan):

    def __init__(self, token_sequence, subtoken_indexes, level: str, spacy_span: Span=None):
        super().__init__(token_sequence, subtoken_indexes, spacy_span=spacy_span)
        self.level = level

    def get_subtype(self):
        return self.level



