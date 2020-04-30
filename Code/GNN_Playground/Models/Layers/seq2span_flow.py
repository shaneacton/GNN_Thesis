from torch import nn
from torch.nn import Linear, LSTM


class Seq2SpanFlow(nn.Module):

    """
    Input:
        designed to take in an attended seq vec, as well as a modeled seq vec
        the modeled vec is the attended vec passed through a modeling layer
        this way the attended vec flows through the modeling layer to
        the output layer

    Output:
        outputs a probability distribution over the input sequence for both the
        start and end index of the predicted span
    """

    def __init__(self, attended_size, hidden_size, dropout):
        super().__init__()
        self.p1_weight_att = Linear(attended_size, 1)
        self.p1_weight_mod = Linear(hidden_size * 2, 1)
        self.p2_weight_att = Linear(attended_size, 1)
        self.p2_weight_mod = Linear(hidden_size * 2, 1)

        self.output_LSTM = LSTM(input_size=hidden_size * 2,
                                hidden_size=hidden_size,
                                bidirectional=True,
                                batch_first=True,
                                dropout=dropout)

    def forward(self, attended_vec, modeled_vec):
        """
        :param attended_vec: (batch, c_len, embedding_dim * 4)
        :param modeled_vec: (batch, c_len ,hidden_size * 2)
        :return: p1: (batch, c_len), p2: (batch, c_len)
        p1, p2 are the start and end index distributions respectively
        """
        # (batch, c_len)
        p1 = (self.p1_weight_att(attended_vec) + self.p1_weight_mod(modeled_vec)).squeeze()
        # (batch, c_len, hidden_size * 2)
        m2 = self.output_LSTM(modeled_vec)[0]
        # (batch, c_len)
        p2 = (self.p2_weight_att(attended_vec) + self.p2_weight_mod(m2)).squeeze()

        return p1, p2