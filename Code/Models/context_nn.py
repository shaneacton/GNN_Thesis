from typing import Dict

from torch import nn

from Code.GNNs import CandidateSelection
from Code.GNNs import OutputModel
from Code.GNNs import SpanSelection
from Code.Training import device


class ContextNN(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.output_model: OutputModel = None

    def init_output_model(self, example: Dict, in_features):
        out_type = self.get_output_model_type(example)
        self.output_model = out_type(in_features).to(device)

    def get_output_model_type(self, example: Dict):
        if 'candidates' in example:
            return CandidateSelection
        else:
            return SpanSelection
