from Code.Config import graph_construction_config
from Code.Data.Text.Tokenisation.document_extract import DocumentExtract


class EntitySpan(DocumentExtract):

    """
        an entity exists with respect to a token sequence
        it may span multiple tokens and subtokens

        an entity can either be a named reference, a coreference
        or a candidate entity wrt: a multiple choice question
    """

    def __init__(self, token_sequence, subtoken_indexes, spacy_span, is_coref=False):
        super().__init__(token_sequence, subtoken_indexes, graph_construction_config.WORD, spacy_span)
        self.is_coref = is_coref

    def __repr__(self):
        text = self.spacy_span.text
        coref = "(coref)" if self.is_coref else ""
        label = ("(" + self.spacy_span.label_ + ")") if self.spacy_span.label_ else ""
        span = ":S" + repr(self.subtoken_indexes)
        return text + coref + label + span

    def get_subtype(self):
        return graph_construction_config.COREF if self.is_coref else graph_construction_config.ENTITY


