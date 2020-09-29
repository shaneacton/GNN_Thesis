from Code.Config import graph_construction_config
from Code.Data.Text.Tokenisation.document_extract import DocumentExtract


class EntitySpan(DocumentExtract):

    """
        an entity exists with respect to a token sequence
        it may span multiple tokens and subtokens

        an entity can either be a named reference, a coreference
        or a candidate entity wrt: a multiple choice question
    """

    def __init__(self, token_sequence, subtoken_indexes, is_coref=False):
        if is_coref:
            raise Exception()
        super().__init__(token_sequence, subtoken_indexes, graph_construction_config.WORD)
        self.is_coref = is_coref

    def __repr__(self):
        coref = "(coref)" if self.is_coref else ""
        word = self.token_sequence.text_obj.raw_text
        span = ":S" + repr(self.subtoken_indexes)
        return coref + word + span

    def get_subtype(self):
        return graph_construction_config.COREF if self.is_coref else graph_construction_config.ENTITY


