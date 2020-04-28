from torch import nn

from Code.GNN_Playground.Data.context import Context
from Code.GNN_Playground.Data.question import Question
from Code.GNN_Playground.Data.training_example import TrainingExample
from Code.GNN_Playground.Models import embedded_size, embedder
from allennlp.modules.attention import DotProductAttention


class AttBiGRU(nn.Module):

    def __init__(self, hidden_size):
        super(AttBiGRU, self).__init__()
        self.context_bi_gru = nn.GRU(embedded_size, hidden_size, num_layers=1,bidirectional=True)
        self.query_bi_gru = nn.GRU(embedded_size, hidden_size, num_layers=1,bidirectional=True)

        self.

    def forward(self, context : Context, question: Question):
        context_embedding = embedder(context.get_full_context())
        query_embedding = embedder(question.text)

        print("context shape:",context_embedding.size(),"query shape:",query_embedding.size())

