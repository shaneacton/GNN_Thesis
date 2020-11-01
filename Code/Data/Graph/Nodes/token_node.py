from transformers import TokenSpan

import Code.constants
from Code import constants
from Code.Data.Graph.Nodes.span_node import SpanNode


class TokenNode(SpanNode):

    def __init__(self, span: TokenSpan, source=Code.constants.CONTEXT):
        super().__init__(span, subtype=constants.TOKEN, source=source)

    def get_structure_level(self):
        return Code.constants.TOKEN
