import torch
from torch import nn
from torch.nn import Linear, Dropout, LayerNorm, ModuleList

from Code.GNNs.gated_gnn import GatedGNN
from Config.config import conf


class GNNStack(nn.Module):

    def __init__(self, GNNClass, use_gating=False, **layer_kwargs):
        super().__init__()
        layers = []
        if "dropout" not in layer_kwargs:
            init_args = GNNClass.__init__.__code__.co_varnames
            print("init args:", init_args)
            if "BASE_GNN_CLASS" in init_args:
                base_args = layer_kwargs["BASE_GNN_CLASS"].__init__.__code__.co_varnames
                if "dropout" in init_args:
                    layer_kwargs.update({"dropout": conf.dropout})
            elif "dropout" in init_args:
                layer_kwargs.update({"dropout": conf.dropout})
        for layer_i in range(conf.num_layers):
            in_size = conf.embedded_dims if layer_i == 0 else conf.hidden_size
            layer = GNNLayer(GNNClass, in_size, **layer_kwargs)
            if use_gating:
                layer = GatedGNN(layer)
            layers.append(layer)

        self.layers = ModuleList(layers)

    def forward(self, x, **kwargs):
        for layer in self.layers:
            x = layer(x, **kwargs)
        return x


class GNNLayer(nn.Module):

    def __init__(self, GNNClass, in_channels, intermediate_fac=2, **layer_kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = conf.hidden_size
        if "heads" in layer_kwargs:
            self.gnn = GNNClass(in_channels, conf.hidden_size//layer_kwargs["heads"], **layer_kwargs)
        else:
            self.gnn = GNNClass(in_channels, conf.hidden_size, **layer_kwargs)

        self.linear1 = Linear(conf.hidden_size, conf.hidden_size * intermediate_fac)
        self.dropout = Dropout(conf.dropout)
        self.linear2 = Linear(conf.hidden_size * intermediate_fac, conf.hidden_size)

        self.norm1 = LayerNorm(conf.hidden_size)
        self.norm2 = LayerNorm(conf.hidden_size)
        self.dropout1 = Dropout(conf.dropout)
        self.dropout2 = Dropout(conf.dropout)

    def forward(self, x, **kwargs):
        "x ~ (N, in_channels)"
        x2 = self.dropout1(self.gnn(x, **kwargs))  # # (N, out_channels)
        if x.size(-1) == x2.size(-1):
            x = x + x2  # residual
            x = self.norm1(x)
        else:  # no residual if this layer is changing the model dim
            x = self.norm1(x2)
        x2 = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        x = x + self.dropout2(x2)  # residual
        x = self.norm2(x)

        return x