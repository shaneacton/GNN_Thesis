from spacy.tokens.span import Span

from Code.Data.Text.Tokenisation.token_span import TokenSpan


class DocumentExtract(TokenSpan):

    Token = "token"
    WORD = "word"
    SENTENCE = "sentence"
    PASSAGE = "passage"
    DOC = "doc"

    levels = [Token, WORD, SENTENCE, PASSAGE, DOC]

    def __init__(self, token_sequence, token_indexes, level: str, spacy_span: Span=None):
        super().__init__(token_sequence, token_indexes, spacy_span=spacy_span)
        self.level = level

    def get_subtype(self):
        return self.level



