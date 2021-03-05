from torch import nn
from torch.nn import TransformerEncoderLayer, LayerNorm, TransformerEncoder

from Code.HDE.Transformers.transformer import Transformer
from Code.HDE.switch_module import SwitchModule
from Config.config import conf


class SwitchTransformer(nn.Module):

    def __init__(self, hidden_size, types=None, intermediate_fac=2, include_global=False):
        super().__init__()
        self.num_heads = conf.heads
        self.num_types = len(types)
        self.hidden_size = hidden_size
        self.include_global = include_global

        encoder_layer = TransformerEncoderLayer(self.hidden_size, conf.transformer_heads,
                                                self.hidden_size * intermediate_fac, conf.dropout, 'relu')
        encoder_norm = LayerNorm(self.hidden_size)
        encoder = TransformerEncoder(encoder_layer, conf.num_transformer_layers, encoder_norm)

        self.switch_encoder = SwitchModule(encoder, types=types, include_global=include_global)

    def get_type_tensor(self, type, length, type_map):
        ids = Transformer.get_type_ids(type, length, type_map)
        return self.type_embedder(ids).view(1, -1, self.hidden_size)
