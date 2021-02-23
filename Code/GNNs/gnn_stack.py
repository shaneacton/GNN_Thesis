import torch
from torch import nn
from torch.nn import Linear, Dropout, LayerNorm, ModuleList

from Code.Config.config import config


class GNNStack(nn.Module):

    def __init__(self, GNNClass, **layer_kwargs):
        super().__init__()
        layers = []
        for layer_i in range(config.num_layers):
            in_size = config.embedded_dims if layer_i == 0 else config.hidden_size
            layer = GNNLayer(GNNClass, in_size, **layer_kwargs)
            layers.append(layer)

        self.layers = ModuleList(layers)

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


class GNNLayer(nn.Module):

    def __init__(self, GNNClass, in_channels, intermediate_fac=2, **layer_kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = config.hidden_size
        self.gnn = GNNClass(in_channels, config.hidden_size//config.heads, heads=config.heads, **layer_kwargs)

        self.linear1 = Linear(config.hidden_size, config.hidden_size * intermediate_fac)
        self.dropout = Dropout(config.dropout)
        self.linear2 = Linear(config.hidden_size * intermediate_fac, config.hidden_size)

        self.norm1 = LayerNorm(config.hidden_size)
        self.norm2 = LayerNorm(config.hidden_size)
        self.dropout1 = Dropout(config.dropout)
        self.dropout2 = Dropout(config.dropout)

    def forward(self, x, edge_index):
        "x ~ (N, in_channels)"
        x2 = self.dropout1(self.gnn(x, edge_index))  # # (N, out_channels)
        if x.size(-1) == x2.size(-1):
            x = x + x2  # residual
            x = self.norm1(x)
        else:  # no residual if this layer is changing the model dim
            x = self.norm1(x2)
        x2 = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        x = x + self.dropout2(x2)  # residual
        x = self.norm2(x)

        return x