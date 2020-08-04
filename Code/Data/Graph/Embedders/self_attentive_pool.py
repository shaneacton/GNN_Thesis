import torch
from torch import nn

from Code.Data.Graph.Embedders.sequence_summariser import SequenceSummariser
from Code.Training import device


class SelfAttentivePool(SequenceSummariser):

    """uses additive attention"""
    # todo implement scaled dot product attention
    # todo multiheaded

    def __init__(self, num_layers):
        # reduces feature dims to 1 over num_layers linear layers. used to do weighted sum over sequence elements
        super().__init__()
        self.num_layers = num_layers
        self.attention_scorer: nn.Sequential = None
        self.softmax = None

    def _init_layers(self, feature_size):
        layer_sizes = []

        def linear_interp(x):
            m = -1 / self.num_layers
            return m * x + 1

        for i in range(self.num_layers + 1):
            size = linear_interp(i) * feature_size
            size = max(int(size), 1)  # last layer maps to 1 feature - the attention score
            layer_sizes.append(size)

        layers = [nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(self.num_layers)]
        self.softmax = nn.Softmax(dim=1)
        self.attention_scorer = nn.Sequential(*layers).to(device)

    def _summarise(self, embedded_sequence: torch.Tensor):
        print("using sap to summ ", embedded_sequence.size())
        scale_fac = 1/pow(embedded_sequence.size(2), 0.5)
        attention_scores = self.attention_scorer(embedded_sequence)
        attention_scores = self.softmax(attention_scores)
        print("att scores:", attention_scores.size(), attention_scores[0,:,0])
        weighted_sequence = attention_scores * embedded_sequence
        # print("embedded sequence:",embedded_sequence[0,0,:])
        # print("weighted_sequence:",weighted_sequence[0,0,:])
        sum = torch.sum(weighted_sequence, dim=1)
        print("sum:",sum.size())
        return sum.view(1, 1, -1)
