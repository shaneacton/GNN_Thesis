from torch import nn

from Code.Data.Text.Answers.extracted_answer import ExtractedAnswer
from Code.Data.Text.data_sample import DataSample
from Code.Models.GNNs.OutputModules.output_model import OutputModel
from Code.Models.GNNs.OutputModules.span_selection import SpanSelection
from Code.Training import device


class ContextNN(nn.Module):

    def __init__(self):
        nn.Module.__init__(self)
        self.output_model: OutputModel = None

    def init_output_model(self, data_sample: DataSample, in_features):
        out_type = self.get_output_model_type(data_sample)
        self.output_model = out_type(in_features).to(device)

    def get_output_model_type(self, data_sample: DataSample):
        answer_type = data_sample.get_answer_type()
        if answer_type == ExtractedAnswer:
            return SpanSelection