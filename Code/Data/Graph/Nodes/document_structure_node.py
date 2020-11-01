from transformers import TokenSpan

import Code.constants
from Code.Data.Graph.Nodes.span_node import SpanNode


class DocumentStructureNode(SpanNode):

    """can represent a sentence, passage, or whole document"""

    def __init__(self, span: TokenSpan, source=Code.constants.CONTEXT, subtype=None):
        """
        :param subtype: can be SENTENCE, PASSAGE, DOCUMENT
        """
        if not subtype:
            raise Exception("must provide a subtype to structure nodes")
        super().__init__(span, subtype=subtype, source=source)

    # def get_node_viz_text(self):
    #     if self.token_span.level == Code.constants.SENTENCE:
    #         return super().get_node_viz_text()
    #     if self.token_span.level == Code.constants.QUERY_SENTENCE:
    #         text = "QUERY: " + self.token_span.text
    #         return "\n".join(textwrap.wrap(text, 16))
    #
    #     if self.token_span.level == Code.constants.TOKEN:
    #         raise Exception()
    #     return repr(self.token_span.level)

    def get_structure_level(self):
        return self.subtype


