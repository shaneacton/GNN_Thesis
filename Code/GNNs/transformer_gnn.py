from torch import nn
from torch.nn import TransformerEncoderLayer, LayerNorm, TransformerEncoder

from Config.config import conf


class TransformerGNN(nn.Module):

    def __init__(self, heads, num_layers, **kwargs):
        super().__init__()
        size = conf.hidden_size
        trans = TransformerEncoderLayer(size, heads,
                                                size * 2, conf.dropout, 'relu')
        encoder_norm = LayerNorm(size)
        self.encoder = TransformerEncoder(trans, num_layers, encoder_norm)

    def forward(self, x, mask, **kwargs):
        x = self.encoder(x.view(x.size(0), 1, -1), mask=mask)
        return x
