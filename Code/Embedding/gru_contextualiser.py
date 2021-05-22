from torch import nn

from Code.Training import dev
from Config.config import conf


class GRUContextualiser(nn.Module):

    """a convienience wrapper around a bi-GRU"""

    def __init__(self, layers=1, dims=None):
        super().__init__()
        self.n_layers = layers
        if dims is None:
            self.hidden_dim = conf.embedded_dims
        else:
            self.hidden_dim = dims

        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim//2, layers, batch_first=True, dropout=conf.dropout,
                          bidirectional=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        no_batch = False
        if len(x.size()) == 2:
            no_batch = True
            x = x.view(1, x.size(0), -1)
        out, h = self.gru(x, self.init_hidden(1))
        assert x.size() == out.size()
        if no_batch:
            out = out.view(x.size(1), -1)
        return self.relu(out)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(2 * self.n_layers, batch_size, self.hidden_dim//2).zero_().to(dev())
        return hidden