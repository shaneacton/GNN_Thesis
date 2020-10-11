from Code.Models.GNNs.OutputModules.output_model import OutputModel
from Code.Models.GNNs.OutputModules.token_selection import TokenSelection


class SpanSelection(OutputModel):

    def __init__(self, in_features):
        super().__init__(in_features)
        self.start_selector = TokenSelection(in_features)
        self.end_selector = TokenSelection(in_features)

    def forward(self, data):
        data.x = self.start_selector(data, inplace=False), self.end_selector(data, inplace=False)
        return data