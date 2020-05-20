from Code.Data.Text.Tokenisation.token_span import TokenSpan
from Code.Models import tail_concatinator


class Entity(TokenSpan):

    """
        an entity exists with respect to a token sequence
        it may span multiple tokens and subtokens

        an entity can either be a named reference, a coreference
        or a candidate entity wrt: a multiple choice question
    """

    def __init__(self, token_sequence, token_span, spacy_span, is_coref=False):
        super().__init__(token_sequence, token_span, spacy_span)
        self.is_coref = is_coref

    def __repr__(self):
        text = self.spacy_span.text
        coref = "(coref)" if self.is_coref else ""
        label = ("(" + self.spacy_span.label_ + ")") if self.spacy_span.label_ else ""
        span = ":S" + repr(self.token_span)
        return text + coref + label + span

    def get_embedding(self, sequence_reduction=tail_concatinator):
        return super().get_embedding(sequence_reduction)


