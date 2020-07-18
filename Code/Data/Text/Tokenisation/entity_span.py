from Code.Data.Text.Tokenisation.document_extract import DocumentExtract
from Code.Models import tail_concatinator


class EntitySpan(DocumentExtract):

    """
        an entity exists with respect to a token sequence
        it may span multiple tokens and subtokens

        an entity can either be a named reference, a coreference
        or a candidate entity wrt: a multiple choice question
    """

    ENTITY = "entity"
    COREF = "coref"

    def __init__(self, token_sequence, token_indexes, spacy_span, is_coref=False):
        super().__init__(token_sequence, token_indexes, DocumentExtract.WORD, spacy_span)
        self.is_coref = is_coref

    def __repr__(self):
        text = self.spacy_span.text
        coref = "(coref)" if self.is_coref else ""
        label = ("(" + self.spacy_span.label_ + ")") if self.spacy_span.label_ else ""
        span = ":S" + repr(self.token_indexes)
        return text + coref + label + span

    def get_embedding(self, sequence_reduction=tail_concatinator):
        return super().get_embedding(sequence_reduction)

    def get_subtype(self):
        return self.COREF if self.is_coref else self.ENTITY


