from torch import Tensor

from Code.Models.GNNs.OutputModules.output_model import OutputModel


class SpanSelection(OutputModel):

    def __init__(self, in_features):
        super().__init__(in_features)
        from Code.Models.GNNs.OutputModules.token_selection import TokenSelection

        self.start_selector = TokenSelection(in_features)
        self.end_selector = TokenSelection(in_features)

    def get_output_from_graph_encoding(self, data, **kwargs):
        return self.start_selector(data, **kwargs), self.end_selector(data, **kwargs)

    def get_output_from_tensor(self, x: Tensor, **kwargs):
        return self.start_selector(x, **kwargs), self.end_selector(x, **kwargs)


