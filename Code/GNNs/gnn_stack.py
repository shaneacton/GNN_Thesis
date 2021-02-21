import torch
from torch import nn
from torch.nn import Linear, Dropout, LayerNorm, ModuleList


class GNNStack(nn.Module):

    def __init__(self, GNNClass, num_layers, in_channels, hidden_size, dropout=0.1, **layer_kwargs):
        super().__init__()
        layers = []
        for layer_i in range(num_layers):
            in_size = in_channels if layer_i == 0 else hidden_size
            layer = GNNLayer(GNNClass, in_size, hidden_size, dropout=dropout, **layer_kwargs)
            layers.append(layer)

        self.layers = ModuleList(layers)

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


class GNNLayer(nn.Module):

    def __init__(self, GNNClass, in_channels, hidden_size, dropout=0.1, **layer_kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.gnn = GNNClass(in_channels, hidden_size, **layer_kwargs)

        self.linear1 = Linear(hidden_size, hidden_size * 2)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(hidden_size * 2, hidden_size)

        self.norm1 = LayerNorm(hidden_size)
        self.norm2 = LayerNorm(hidden_size)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, x, edge_index):
        "x ~ (N, in_channels)"
        x2 = self.dropout1(self.gnn(x, edge_index))  # # (N, out_channels)
        if self.hidden_size == self.in_channels:
            x = x + x2  # residual
            x = self.norm1(x)
        else:  # no residual if this layer is changing the model dim
            x = self.norm1(x2)
        x2 = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        x = x + self.dropout2(x2)  # residual
        x = self.norm2(x)

        return x