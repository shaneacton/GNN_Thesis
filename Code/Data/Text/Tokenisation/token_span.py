from typing import Tuple


class TokenSpan:

    """
        a tokenspan exists with respect to a token sequence
        it may span multiple tokens and subtokens
    """

    def __init__(self, token_sequence, subtoken_indexes):
        # todo candidate subclass
        self.token_sequence = token_sequence
        self.subtoken_indexes: Tuple[int] = subtoken_indexes
        self.level = type(self)

    @property
    def text(self):
        return self.token_sequence.get_text_from_subspan(self.subtoken_indexes)

    @property
    def tokens(self):
        word_token_span = self.token_sequence.convert_subtoken_span_to_word_token_span(self.subtoken_indexes)
        return self.token_sequence.raw_word_tokens[word_token_span[0]: word_token_span[1]]

    @property
    def subtokens(self):
        return self.token_sequence.raw_subtokens[self.subtoken_indexes[0]: self.subtoken_indexes[1]]

    def __repr__(self):
        span = ":S" + repr(self.subtoken_indexes)
        return repr(type(self)) + " " + self.text + span

    def span_match(self, other):
        return self.token_sequence == other.token_sequence and self.subtoken_indexes == other.subtoken_indexes

    def __eq__(self, other):
        return self.span_match(other) and type(self) == type(other)

    def __hash__(self):
        return self.token_sequence.__hash__() + 7 * self.subtoken_indexes.__hash__() + 3 * hash(type(self))

    def distance(self, other_entity):
        """
        returns the distance in tokens between these two entities
        """
        if self.token_sequence != other_entity.token_sequence:
            raise Exception("cannot calculate distance of two entities not in the same text")
        dist = self.subtoken_indexes[1] - other_entity.subtoken_indexes[0]
        dist = max(dist, other_entity.subtoken_indexes[1] - self.subtoken_indexes[0])
        return dist - 1

    def contains(self, other_span):
        contains_start = self.subtoken_indexes[0] <= other_span.subtoken_indexes[0] < self.subtoken_indexes[1]
        contains_end = self.subtoken_indexes[0] < other_span.subtoken_indexes[1] <= self.subtoken_indexes[1]
        if contains_end != contains_start:
            raise Exception("overlapping spans- " + repr(self) + "\n" + repr(other_span))
        return contains_start and contains_end

    def before(self, other_span):
        """returns true if self preceeds other in the token seq"""
        # todo check for overlaps
        return self.subtoken_indexes[1] <= other_span.subtoken_indexes[0]

