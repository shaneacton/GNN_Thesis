from typing import List

from spacy.tokens.span import Span

from Code.Data.Text.Tokenisation.entity import Entity
from Code.Data.Text.Tokenisation.token_span import TokenSpan


class DocumentExtract(TokenSpan):

    SENTENCE = "sentence"
    PASSAGE = "passage"
    DOC = "doc"

    def __init__(self, token_sequence, token_span, type: str,spacy_span: Span=None):
        super().__init__(token_sequence, token_span, spacy_span=spacy_span)
        self.contained_entities: List[Entity] = []
        self.type = type

    def add_entity(self, ent:Entity):
        self.contained_entities.append(ent)




