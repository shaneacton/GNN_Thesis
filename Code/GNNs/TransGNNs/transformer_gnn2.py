from torch import nn

from Code.Transformers.PytorchReimpl.MultiheadAttentionRe import MultiheadAttentionRe
from Config.config import conf


class TransformerGNN2(nn.Module):

    def __init__(self, in_channels, out_channels, heads, **kwargs):
        super().__init__()
        if in_channels != out_channels:
            raise Exception("")
        self.self_attn = MultiheadAttentionRe(in_channels, heads, dropout=conf.dropout, batch_first=True)

    def forward(self, x, mask, **kwargs):
        """x~(l,f)"""
        x = x.view(1, -1, x.size(-1))
        print("transgnn2. x:", x.size())
        x = self.self_attn(x, x, x, attn_mask=mask, key_padding_mask=None)[0].view(-1, x.size(-1))
        print("out:", x.size())
        return x
