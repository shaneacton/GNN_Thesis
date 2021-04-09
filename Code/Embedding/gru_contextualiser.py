from torch import nn

from Code.Training import dev
from Config.config import conf


class GRUContextualiser(nn.Module):

    def __init__(self, layers=1):
        super().__init__()
        self.n_layers = layers
        self.hidden_dim = conf.embedded_dims
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim//2, layers, batch_first=True, dropout=conf.dropout,
                          bidirectional=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, h = self.gru(x, self.init_hidden(1))
        assert x.size() == out.size()
        return self.relu(out)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(2 * self.n_layers, batch_size, self.hidden_dim//2).zero_().to(dev())
        return hidden