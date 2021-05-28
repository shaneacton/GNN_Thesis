from torch import nn
from torch.nn import TransformerEncoderLayer, LayerNorm, TransformerEncoder

from Config.config import conf


class TransformerGNN(nn.Module):

    def __init__(self, hidden_size, heads, num_layers, **kwargs):
        super().__init__()
        trans = TransformerEncoderLayer(hidden_size, heads,
                                                hidden_size * 2, conf.dropout, 'relu')
        encoder_norm = LayerNorm(hidden_size)
        self.encoder = TransformerEncoder(trans, num_layers, encoder_norm)

    def forward(self, x, mask, **kwargs):
        x = self.encoder(x.view(x.size(0), 1, -1), mask=mask)
        return x
