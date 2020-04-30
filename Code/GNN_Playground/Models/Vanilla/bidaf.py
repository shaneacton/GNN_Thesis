import torch.nn as nn
from torch.nn import Linear, LSTM

from Code.GNN_Playground.Data.context import Context
from Code.GNN_Playground.Data.question import Question
from Code.GNN_Playground.Models import embedded_size
from Code.GNN_Playground.Models.Layers.attention_flow import AttentionFlow
from Code.GNN_Playground.Models.Layers.seq2span_flow import Seq2SpanFlow

"""
    source: https://github.com/galsang/BiDAF-pytorch
    author: Taeuk Kim, Ph.D. student, Seoul National University
    
    code has been modified to use projects main text encoder instead of the char + word
    embedding and contextualisation used by the original implementation
    thus parts 1-3 of the BiDaf model have been removed
"""

class BiDAF(nn.Module):
    def __init__(self, hidden_size, dropout=0.2):
        super(BiDAF, self).__init__()
        # 4. Attention Flow Layer
        self.att_flow_layer = AttentionFlow()

        # 5. Modeling Layer
        self.modeling_LSTM1 = LSTM(input_size=embedded_size * 4,
                                   hidden_size=hidden_size,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=dropout)

        self.modeling_LSTM2 = LSTM(input_size=hidden_size * 2,
                                   hidden_size=hidden_size,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=dropout)

        # 6. Output Layer
        self.output_layer = Seq2SpanFlow(embedded_size * 4, hidden_size, dropout)

    def forward(self, context: Context, query: Question):
        """
            context and query are vecs of (batch, seq, embedding_dim)
        """
        # 1. Attention Flow Layer
        attended_vec = self.att_flow_layer(context.get_context_embedding(), query.get_embedding())
        # attended_vec ~ (batch, context_seq_len, embedding_dim * 4)
        # 2. Modeling Layer
        modeled_vec = self.modeling_LSTM2(self.modeling_LSTM1(attended_vec)[0])[0]
        # modeled_vec ~  (batch, context_seq_len, hidden_size * 2)
        # 3. Output Layer
        p1, p2 = self.output_layer(attended_vec, modeled_vec)
        # (batch, c_len), (batch, c_len)
        return p1, p2