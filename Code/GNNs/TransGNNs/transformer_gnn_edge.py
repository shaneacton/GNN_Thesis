from torch import nn

from Code.Transformers.PytorchReimpl.MultiheadAttentionEdge import MultiheadAttentionEdge
from Config.config import conf


class TransformerGNNEdge(nn.Module):

    def __init__(self, in_channels, out_channels, heads, **kwargs):
        super().__init__()
        if in_channels != out_channels:
            raise Exception("")
        self.self_attn = MultiheadAttentionEdge(in_channels, heads, 14, dropout=conf.dropout, batch_first=True)

    def forward(self, x, mask, **kwargs):
        """x~(l,f)"""
        x = x.view(1, -1, x.size(-1))
        x = self.self_attn(x, x, x, attn_mask=mask, key_padding_mask=None, **kwargs)[0].view(-1, x.size(-1))
        return x
