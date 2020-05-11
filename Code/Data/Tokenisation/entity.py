from typing import Tuple

from spacy.tokens.span import Span


class Entity:

    """
        an entity exists with respect to a token sequence
        it may span multiple tokens and subtokens
    """

    def __init__(self, spacy_entity: Span, token_span):
        self.spacy_entity : Span = spacy_entity
        self.token_span: Tuple[int] = token_span

    def __repr__(self):
        return self.spacy_entity.text + "(" + self.spacy_entity.label_ + ")" + ":S" + repr(self.token_span)