import torch.nn as nn
from torch.nn import Linear, LSTM

from Code.GNN_Playground.Models import embedded_size
from Code.GNN_Playground.Models.Layers.attention_flow import AttentionFlow

"""
    source: https://github.com/galsang/BiDAF-pytorch
    
    code has been modified to projects main text encoder instead of the char + word
    embedding and contextualisation used by the original implementation
    thus parts 1-3 of the BiDaf model have been removed
"""

class BiDAF(nn.Module):
    def __init__(self, hidden_size, args, pretrained):
        super(BiDAF, self).__init__()
        self.args = args

        # 4. Attention Flow Layer
        self.att_flow_layer = AttentionFlow()

        # 5. Modeling Layer
        self.modeling_LSTM1 = LSTM(input_size=embedded_size * 4,
                                   hidden_size=hidden_size,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=args.dropout)

        self.modeling_LSTM2 = LSTM(input_size=hidden_size * 2,
                                   hidden_size=hidden_size,
                                   bidirectional=True,
                                   batch_first=True,
                                   dropout=args.dropout)

        # 6. Output Layer
        self.p1_weight_g = Linear(embedded_size * 4, 1)
        self.p1_weight_m = Linear(hidden_size * 2, 1)
        self.p2_weight_g = Linear(embedded_size * 4, 1)
        self.p2_weight_m = Linear(hidden_size * 2, 1)

        self.output_LSTM = LSTM(input_size=hidden_size * 2,
                                hidden_size=hidden_size,
                                bidirectional=True,
                                batch_first=True,
                                dropout=args.dropout)

        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, contex, query):
        """
        context and query are vecs of (batch, seq, embedding_dim)
        """

        def output_layer(g, m):
            """
            :param g: (batch, c_len, embedding_dim * 4)
            :param m: (batch, c_len ,hidden_size * 2)
            :return: p1: (batch, c_len), p2: (batch, c_len)
            """
            # (batch, c_len)
            p1 = (self.p1_weight_g(g) + self.p1_weight_m(m)).squeeze()
            # (batch, c_len, hidden_size * 2)
            m2 = self.output_LSTM(m)[0]
            # (batch, c_len)
            p2 = (self.p2_weight_g(g) + self.p2_weight_m(m2)).squeeze()

            return p1, p2

        # 4. Attention Flow Layer
        g = self.att_flow_layer(contex, query)
        # g ~ (batch, context_seq_len, embedding_dim * 4)
        # 5. Modeling Layer
        m = self.modeling_LSTM2(self.modeling_LSTM1(g)[0])[0]
        # m ~  (batch, context_seq_len, hidden_size * 2)
        # 6. Output Layer
        p1, p2 = output_layer(g, m)

        # (batch, c_len), (batch, c_len)
        return p1, p2