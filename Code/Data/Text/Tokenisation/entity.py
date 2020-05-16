from typing import Tuple

from spacy.tokens.span import Span

from Code.Models import tail_concatinator


class Entity:

    """
        an entity exists with respect to a token sequence
        it may span multiple tokens and subtokens

        an entity can either be a named reference, a coreference
        or a candidate entity wrt: a multiple choice question
    """

    def __init__(self, spacy_entity: Span, token_sequence, token_span, is_coref=False, is_candidate=False):
        # todo candidate subclass
        self.spacy_entity : Span = spacy_entity
        self.token_sequence = token_sequence
        self.token_span: Tuple[int] = token_span
        self.is_coref = is_coref
        self.is_candidate = is_candidate

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
        text = self.spacy_entity.text
        coref = "(coref)" if self.is_coref else ""
        label = ("(" + self.spacy_entity.label_ + ")") if self.spacy_entity.label_ else ""
        span = ":S" + repr(self.token_span)
        return text + coref + label + span

    def __eq__(self, other):
        return self.token_sequence == other.token_sequence and self.token_span == other.token_span

    def __hash__(self):
        return self.token_sequence.__hash__() + 7* self.token_span.__hash__()

    def distance(self, other_entity):
        """
        returns the distance in tokens between these two entities
        """
        if self.token_sequence != other_entity.token_sequence:
            raise Exception("cannot calculate distance of two entities not in the same text")
        dist = self.token_span[1] - other_entity.token_span[0]
        dist = max(dist, other_entity.token_span[1] - self.token_span[0])
        return dist - 1

    def get_embedding(self, sequence_reduction=tail_concatinator):
        full_embedding = self.token_sequence.text.full_embedding
        span = self.subtoken_span
        entity_embedding = full_embedding[:,span[0]:span[1],:]
        if sequence_reduction:
            return sequence_reduction(entity_embedding)
        return entity_embedding
