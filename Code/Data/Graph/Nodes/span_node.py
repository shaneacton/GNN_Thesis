import textwrap
from abc import ABC
from typing import Dict

from torch import Tensor

from Code.Data.Graph.Nodes.node import Node
from Code.Data.Graph.State.state_set import StateSet
from Code.Data.Text.Tokenisation.token_span import TokenSpan


class SpanNode(Node, ABC):

    EMB_IDS = "emb_ids"

    def __init__(self, token_span: TokenSpan, subtype=None):
        self.token_span = token_span
        super().__init__(subtype=subtype)

    def get_node_viz_text(self):
        text = self.token_span.text + "\n" + repr(self.token_span.token_indexes)
        return "\n".join(textwrap.wrap(text, 16))

    def __eq__(self, other):
        return self.token_span == other.token_span

    def __hash__(self):
        return hash(self.token_span)

    def get_embedding_ids_sequence_tensor(self) -> Tensor:
        return self.token_span.subtoken_embedding_ids

    def get_all_node_state_tensors(self) -> Dict[str, Tensor]:
        # states = {SpanNode.EMB_IDS: self.get_embedding_ids_sequence_tensor()}  # returns the nodes embedding IDs
        states = {StateSet.STARTING_STATE: self.get_span_summary_vec()}  # gets feature representation of token sequence
        return states

    def get_span_summary_vec(self) -> Tensor:
        """
        returns a tensor with the span representation in a fixed size.
        for entities this summary is done with head/tail concat
        for document structure nodes this summary is done via a separate encoder model
        """
        raise NotImplementedError()
