import textwrap

import Code.constants
from Code.Config import graph_construction_config as construction
from Code.Data.Graph.Nodes.span_node import SpanNode
from Code.Data.Text.Tokenisation.document_extract import DocumentExtract


class DocumentStructureNode(SpanNode):

    """can represent a sentence, passage, or whole document"""

    def __init__(self, document_extract: DocumentExtract, source=Code.constants.CONTEXT, subtype=None):
        super().__init__(document_extract, subtype=subtype, source=source)

        if self.token_span.level == Code.constants.TOKEN:
            raise Exception()

    def get_node_viz_text(self):
        if self.token_span.level == Code.constants.SENTENCE:
            return super().get_node_viz_text()
        if self.token_span.level == Code.constants.QUERY_SENTENCE:
            text = "QUERY: " + self.token_span.text
            return "\n".join(textwrap.wrap(text, 16))

        if self.token_span.level == Code.constants.TOKEN:
            raise Exception()
        return repr(self.token_span.level)

    def get_structure_level(self):
        return self.token_span.level


