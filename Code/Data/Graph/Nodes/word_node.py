from transformers import TokenSpan

import Code.constants
from Code import constants
from Code.Data.Graph.Nodes.span_node import SpanNode


class WordNode(SpanNode):

    def __init__(self, span: TokenSpan, source=Code.constants.CONTEXT, subtype=None):
        """
        :param subtype: can be WORD, ENTITY, UNIQUE_ENTITY
        """
        if not subtype:
            subtype = constants.WORD
        super().__init__(span, subtype=subtype, source=source)

    def get_structure_level(self):
        return Code.constants.WORD
