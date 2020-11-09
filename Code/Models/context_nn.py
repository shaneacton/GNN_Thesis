from typing import Dict

from torch import nn

from Code.Models.GNNs.OutputModules.candidate_selection import CandidateSelection
from Code.Models.GNNs.OutputModules.output_model import OutputModel
from Code.Models.GNNs.OutputModules.span_selection import SpanSelection
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
