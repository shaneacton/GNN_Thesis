from typing import Dict

from torch import Tensor

from Code.Data.Graph.Nodes.span_node import SpanNode
from Code.Data.Text.Tokenisation.document_extract import DocumentExtract


class DocumentStructureNode(SpanNode):

    """can represent a sentence, passage, or whole document"""

    def get_node_states(self) -> Dict[str, Tensor]:
        pass

    def __init__(self, document_extract: DocumentExtract):
        super().__init__(document_extract)

    def get_node_viz_text(self):
        if self.token_span.level == DocumentExtract.SENTENCE:
            return super().get_node_viz_text()
        return repr(self.token_span.level)