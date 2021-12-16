from torch import nn
from torch.nn import TransformerEncoderLayer, LayerNorm, TransformerEncoder

from Config.config import conf


class TransformerGNN(nn.Module):

    def __init__(self, in_channels, out_channels, heads, **kwargs):
        super().__init__()
        if in_channels != out_channels:
            raise Exception("")
        trans = TransformerEncoderLayer(in_channels, heads,
                                                in_channels * 2, conf.dropout, 'relu')
        encoder_norm = LayerNorm(in_channels)
        self.encoder = TransformerEncoder(trans, 1, encoder_norm)

    def forward(self, x, mask, **kwargs):
        x = self.encoder(x.view(x.size(0), 1, -1), mask=mask).view(x.size(0), -1)
        return x
