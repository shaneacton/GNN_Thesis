from torch import Tensor

from Code.Config import configuration
from Code.Data.Graph.Nodes.span_node import SpanNode
from Code.Data.Text.Tokenisation.document_extract import DocumentExtract


class DocumentStructureNode(SpanNode):

    """can represent a sentence, passage, or whole document"""

    def __init__(self, document_extract: DocumentExtract, subtype=None):
        super().__init__(document_extract, subtype=subtype)

    def get_node_viz_text(self):
        if self.token_span.level == configuration.SENTENCE:
            return super().get_node_viz_text()
        return repr(self.token_span.level)