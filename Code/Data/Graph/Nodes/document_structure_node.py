from torch import Tensor

from Code.Data.Graph.Nodes.span_node import SpanNode
from Code.Data.Text.Tokenisation.document_extract import DocumentExtract


class DocumentStructureNode(SpanNode):

    """can represent a sentence, passage, or whole document"""

    def __init__(self, document_extract: DocumentExtract, subtype=None):
        super().__init__(document_extract, subtype=subtype)

    def get_sensory_state(self) -> Tensor:
        # todo better sumary vec
        return self.token_span.tail_concat_embedding

    def get_node_viz_text(self):
        if self.token_span.level == DocumentExtract.SENTENCE:
            return super().get_node_viz_text()
        return repr(self.token_span.level)

    def get_span_summary_vec(self) -> Tensor:
        # todo implement summary encoder
        return self.token_span.tail_concat_embedding