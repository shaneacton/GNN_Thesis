from torch import Tensor

from Code.Data.Graph.graph_encoding import GraphEncoding
from Code.Models.GNNs.OutputModules.output_model import OutputModel
from Code.Models.Loss.loss_funcs import get_span_loss


class SpanSelection(OutputModel):

    def __init__(self, in_features):
        super().__init__(in_features)
        from Code.Models.GNNs.OutputModules.token_selection import TokenSelection

        self.start_selector = TokenSelection(in_features)
        self.end_selector = TokenSelection(in_features)

    def get_output_from_graph_encoding(self, data: GraphEncoding, **kwargs):
        logits = self.start_selector(data, **kwargs), self.end_selector(data, **kwargs)
        if "start_positions" in kwargs and "end_positions" in kwargs:
            positions = kwargs["start_positions"], kwargs["end_positions"]
            loss = get_span_loss(positions[0], positions[1], logits[0], logits[1])

            # print("num prediction options:", logits[0].size(), "ans pos:", positions)
            return (loss,) + logits
        return logits