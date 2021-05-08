import inspect

import torch
from torch import nn
from torch.nn import Linear, Dropout, LayerNorm, ModuleList
from torch_geometric.nn import GATConv

from Code.GNNs.gated_gnn import GatedGNN
from Code.GNNs.wrap_gnn import EdgeEmbeddings
from Config.config import conf


class GNNStack(nn.Module):

    def __init__(self, GNNClass, use_edge_type_embs=False, **layer_kwargs):
        super().__init__()
        self.use_edge_type_embs = use_edge_type_embs
        layers = self.get_layers(GNNClass, layer_kwargs)
        self.layers = ModuleList(layers)
        self.act = None
        if conf.gnn_stack_act != "none":
            if conf.gnn_stack_act == "relu":
                self.act = nn.ReLU()
            else:
                raise Exception("unreckognised activation: " + repr(conf.gnn_stack_act) )

    def get_layers(self, GNNClass, layer_kwargs):
        layers = []
        if "dropout" not in layer_kwargs:
            init_args = GNNClass.__init__.__code__.co_varnames
            #print("init args:", init_args)
            if "BASE_GNN_CLASS" in init_args:
                base_args = layer_kwargs["BASE_GNN_CLASS"].__init__.__code__.co_varnames
                if "dropout" in base_args:
                    layer_kwargs.update({"dropout": conf.dropout})
            elif "dropout" in init_args:
                layer_kwargs.update({"dropout": conf.dropout})
        layer_kwargs.setdefault("aggr", conf.gnn_aggr)
        layer_kwargs.setdefault("add_self_loops", conf.add_self_loops)
        for layer_i in range(conf.num_layers):
            if conf.layerwise_weight_sharing and layer_i > 0:
                """copy by reference so the layers share the same params"""
                layers.append(layers[0])
                break

            in_size = conf.embedded_dims if layer_i == 0 else conf.hidden_size
            LayerWrapper = SimpleGNNLayer if conf.use_simple_gnn else GNNLayer
            layer = LayerWrapper(GNNClass, in_size, use_edge_type_embs=self.use_edge_type_embs, **layer_kwargs)
            if conf.use_gating:
                layer = GatedGNN(layer)

            layers.append(layer)
        return layers

    def forward(self, x, **kwargs):
        res = {}  # residual attention scores
        for l, layer in enumerate(self.layers):
            kwargs.update(res)  # pass previous attention scores forward
            x = layer(x, **kwargs)
            gnn = get_core_gnn(layer)
            if hasattr(gnn, "last_attention_scores") and gnn.last_attention_scores is not None:
                res = {"previous_attention_scores": gnn.last_attention_scores}  # store for next layer

            if self.act is not None:
                x = self.act(x)
        return x


def get_core_gnn(layer):  # unpeels any nested gnns, eg Gate(Rel(Gat(x)))
    gnn = layer.gnn
    while hasattr(gnn, "gnn") or hasattr(gnn, "gnn_layer"):
        gnn = gnn.gnn
        while hasattr(gnn, "gnn_layer"):
            gnn = gnn.gnn_layer
        while hasattr(gnn, "gnn"):
            gnn = gnn.gnn
    return gnn


class GNNLayer(nn.Module):

    def __init__(self, GNNClass, in_channels, intermediate_fac=2, use_edge_type_embs=False, **layer_kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = conf.hidden_size
        init_args = inspect.getfullargspec(GNNClass.__init__)[0]
        needed_kwargs = {k: v for k, v in layer_kwargs.items() if k in init_args}

        if GNNClass == GATConv:
            self.gnn = GNNClass(in_channels, conf.hidden_size//layer_kwargs["heads"], **needed_kwargs)
        else:
            self.gnn = GNNClass(in_channels, conf.hidden_size, **needed_kwargs)

        if use_edge_type_embs:
            num_types = 7 - len(conf.ignored_edges) + 1  # +1 for self edges
            if hasattr(conf, "bidirectional") and conf.bidirectional:
                num_types += 7
            self.gnn = EdgeEmbeddings(self.gnn, in_channels, num_types)

        self.linear1 = Linear(conf.hidden_size, conf.hidden_size * intermediate_fac)
        self.dropout = Dropout(conf.dropout)
        self.linear2 = Linear(conf.hidden_size * intermediate_fac, conf.hidden_size)

        self.norm1 = LayerNorm(conf.hidden_size)
        self.norm2 = LayerNorm(conf.hidden_size)
        self.dropout1 = Dropout(conf.dropout)
        self.dropout2 = Dropout(conf.dropout)

    def forward(self, x, **kwargs):
        """x ~ (N, in_channels)"""
        forward_args = inspect.getfullargspec(self.gnn.forward)[0]
        if "data" in forward_args:
            inp = (x, kwargs.pop("edge_index"))
        else:
            inp = x
        x2 = self.dropout1(self.gnn(inp, **kwargs))  # # (N, out_channels)
        if x.size(-1) == x2.size(-1):
            x = x + x2  # residual
            x = self.norm1(x)
        else:  # no residual if this layer is changing the model dim
            x = self.norm1(x2)
        x2 = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        x = x + self.dropout2(x2)  # residual
        x = self.norm2(x)

        return x
    
    
class SimpleGNNLayer(nn.Module):
    def __init__(self, GNNClass, in_channels, use_edge_type_embs=False, **layer_kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = conf.hidden_size
        if GNNClass == GATConv:
            self.gnn = GNNClass(in_channels, conf.hidden_size//layer_kwargs["heads"], **layer_kwargs)
        else:
            self.gnn = GNNClass(in_channels, conf.hidden_size, **layer_kwargs)

        if use_edge_type_embs:
            num_types = 7 - len(conf.ignored_edges) + 1  # +1 for self edges
            self.gnn = EdgeEmbeddings(self.gnn, in_channels, num_types)

        self.dropout1 = Dropout(conf.dropout)

    def forward(self, x, **kwargs):
        "x ~ (N, in_channels)"
        x = self.dropout1(torch.relu(self.gnn(x, **kwargs)))  # # (N, out_channels)
        return x