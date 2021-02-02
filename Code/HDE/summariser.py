from torch import nn, Tensor
from torch.nn import TransformerEncoderLayer, LayerNorm, TransformerEncoder


class Summariser(nn.Module):
    """
        a summarising longformer which is used to map variable length token embeddings for a node,
        into fixed size node embedding
    """

    def __init__(self, hidden_size, num_layers=1, num_heads=5, intermediate_fac=2, dropout=0.1):
        super().__init__()
        # self.long_conf = get_longformer_config(num_layers=2, num_types=num_types, hidden_size=hidden_size)
        # self.longformer = LongformerModel(self.long_conf)

        encoder_layer = TransformerEncoderLayer(hidden_size, num_heads, hidden_size * intermediate_fac, dropout, 'relu')
        encoder_norm = LayerNorm(hidden_size)
        self.encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        self.hidden_size = hidden_size

    def get_summary_vec(self, full_vec: Tensor, span=None):
        """
            if provided span is the [start,end) indices of the full vec which are to be summarised
            :full_vec ~ (batch, seq_len, features)

            returns (batch, features)
        """
        if not span:
            span = (0, full_vec.size(-2))  # full span
        # else:
        #     print("got span", span, "vec:", full_vec[:, span[0]: span[1], :].size(), "full vec:", full_vec.size())

        vec = full_vec[:, span[0]: span[1], :]
        if vec.size(1) < 1:
            raise Exception("cannot get summary vec, no elements in sequence:" + repr(vec.size()) + " span: " + repr(span)  + " full: " + repr(full_vec.size()))
        # out = self.longformer(inputs_embeds=vec, return_dict=True, output_hidden_states=True)
        embs = self.encoder(vec)
        # embs = out["hidden_states"][-1]  # last hidden
        # print("last hidden states:", embs.size())
        pooled_emb = embs[:, 0]  # the pooled out is the output of the classification token
        # print("pooled size:", pooled_emb.size())

        return pooled_emb

    def forward(self, full_vec: Tensor, span=None):
        return self.get_summary_vec(full_vec, span=span)