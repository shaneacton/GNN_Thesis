from typing import Tuple

from spacy.tokens.span import Span


class TokenSpan:

    """
        a tokenspan exists with respect to a token sequence
        it may span multiple tokens and subtokens
    """

    def __init__(self, token_sequence, token_span, spacy_span: Span=None):
        # todo candidate subclass
        self.spacy_span : Span = spacy_span
        self.token_sequence = token_sequence
        self.token_span: Tuple[int] = token_span
        self.level = type(self)

    @property
    def text(self):
        return self.spacy_span.text if self.spacy_span else self.token_sequence.get_text_from_span(self.token_span)

    @property
    def subtoken_span(self):
        return self.token_sequence.convert_token_span_to_subtoken_span(self.token_span)

    @property
    def tokens(self):
        return self.token_sequence.tokens[self.token_span[0]: self.token_span[1]]

    @property
    def subtokens(self):
        span = self.subtoken_span
        return self.token_sequence.sub_tokens[span[0]: span[1]]

    def __repr__(self):
        text = self.spacy_span.text
        label = ("(" + self.spacy_span.label_ + ")") if self.spacy_span.label_ else ""
        span = ":S" + repr(self.token_span)
        return text  + label + span

    def __eq__(self, other):
        return self.token_sequence == other.token_sequence and self.token_span == other.token_span

    def __hash__(self):
        return self.token_sequence.__hash__() + 7 * self.token_span.__hash__()

    def distance(self, other_entity):
        """
        returns the distance in tokens between these two entities
        """
        if self.token_sequence != other_entity.token_sequence:
            raise Exception("cannot calculate distance of two entities not in the same text")
        dist = self.token_span[1] - other_entity.token_span[0]
        dist = max(dist, other_entity.token_span[1] - self.token_span[0])
        return dist - 1

    def contains(self, other_span):
        contains_start = self.token_span[0] <= other_span.token_span[0] < self.token_span[1]
        contains_end = self.token_span[0] < other_span.token_span[1] <= self.token_span[1]
        if contains_end != contains_start:
            raise Exception("overlapping spans- " + repr(self) + ", " + repr(other_span))
        return contains_start and contains_end

    def before(self, other_span):
        """returns true if self preceeds other in the token seq"""
        # todo check for overlaps
        return self.token_span[1] <= other_span.token_span[0]

    def get_embedding(self, sequence_reduction=None):
        full_embedding = self.token_sequence.text.full_embedding
        span = self.subtoken_span
        entity_embedding = full_embedding[:,span[0]:span[1],:]
        if sequence_reduction:
            return sequence_reduction(entity_embedding)
        return entity_embedding

