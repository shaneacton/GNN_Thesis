from transformers import TokenSpan

from Code.Data.Graph.Nodes.span_node import SpanNode
from Code.constants import SENTENCE, CONTEXT, QUERY


class StructureNode(SpanNode):

    """can represent a sentence, passage, or whole document"""

    def __init__(self, span: TokenSpan, source=CONTEXT, subtype=None):
        """
        :param subtype: can be SENTENCE, PASSAGE, DOCUMENT
        """
        if not subtype:
            raise Exception("must provide a subtype to structure nodes")
        super().__init__(span, subtype=subtype, source=source)

    def get_node_viz_text(self, example):
        prefix = "QUERY: " if self.source == QUERY else ""
        if self.subtype == SENTENCE:
            return prefix + super().get_node_viz_text(example)

        return repr(self.subtype) + ": " + repr(self.token_span)

    def get_structure_level(self):
        return self.subtype


