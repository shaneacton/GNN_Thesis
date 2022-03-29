import copy
import inspect

import torch
from torch import nn
from torch.nn import Linear, Dropout, LayerNorm, ModuleList
from torch_geometric.nn import GATConv

from Code.GNNs.gated_gnn import GatedGNN, SharedGatedGNN
from Code.GNNs.switch_gnn import SwitchGNN
from Code.GNNs.edge_embedding_gnn import EdgeEmbeddings
from Config.config import get_config


def share_gnn(layer, layer0):
    if get_config().use_gating:
        if get_config().use_transformer_block:  # both
            layer.gnn.gnn = layer0.gnn.gnn
        else:  # just gating
            layer.gnn = layer0.gnn
    else:  # no gating
        if get_config().use_transformer_block:  # just trans
            layer.gnn = layer0.gnn
        else:  # neither
            layer = layer0
    return layer


class GNNStack(nn.Module):

    def __init__(self, GNNClass, use_edge_type_embs=False, **layer_kwargs):
        super().__init__()
        self.use_edge_type_embs = use_edge_type_embs
        layers = self.get_layers(GNNClass, layer_kwargs)
        self.layers = ModuleList(layers)
        self.act = None
        if get_config().gnn_stack_act != "none":
            if get_config().gnn_stack_act == "relu":
                self.act = nn.ReLU()
            else:
                raise Exception("unreckognised activation: " + repr(get_config().gnn_stack_act) )

    def get_layers(self, GNNClass, layer_kwargs):
        layers = []
        if "dropout" not in layer_kwargs:
            init_args = GNNClass.__init__.__code__.co_varnames
            #print("init args:", init_args)
            if "BASE_GNN_CLASS" in init_args:
                base_args = layer_kwargs["BASE_GNN_CLASS"].__init__.__code__.co_varnames
                if "dropout" in base_args:
                    layer_kwargs.update({"dropout": get_config().dropout})
            elif "dropout" in init_args:
                layer_kwargs.update({"dropout": get_config().dropout})
        layer_kwargs.setdefault("aggr", get_config().gnn_aggr)
        layer_kwargs.setdefault("add_self_loops", get_config().add_self_loops)

        for layer_i in range(get_config().num_layers):
            if get_config().layerwise_weight_sharing and layer_i > 0:
                """copy by reference so the layers share the same params"""
                layers.append(layers[0])
                continue
            if get_config().share_tuf_params and layer_i > 0 and get_config().use_transformer_block:
                first_layer = layers[0].gnn if get_config().use_gating else layers[0]
                layer = SharedTransGNNLayer(first_layer)
            else:
                LayerWrapper = TransGNNLayer if get_config().use_transformer_block else SimpleGNNLayer
                layer = LayerWrapper(GNNClass, use_edge_type_embs=self.use_edge_type_embs, **layer_kwargs)

            if get_config().use_gating:
                if get_config().share_gate_params and layer_i > 0:
                    layer = SharedGatedGNN(layers[0])
                else:
                    layer = GatedGNN(layer)

            if get_config().share_gnn_params and layer_i > 0:
                layer = share_gnn(layer, layers[0])

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
        while hasattr(gnn, "gnn_layer"):
            gnn = gnn.gnn_layer
        while hasattr(gnn, "gnn"):
            gnn = gnn.gnn
    return gnn


class TUF(nn.Module):
    def __init__(self, size, gnn, linear1, linear2):
        super().__init__()
        self.size = size
        self.gnn = gnn
        self.linear1 = linear1
        self.dropout = Dropout(get_config().dropout)
        self.linear2 = linear2

        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.dropout1 = Dropout(get_config().dropout)
        self.dropout2 = Dropout(get_config().dropout)

    def forward(self, x, *inputs, **kwargs):
        """x ~ (N, in_channels)"""
        forward_args = inspect.getfullargspec(self.gnn.forward)[0]
        if "data" in forward_args:
            inp = (x, kwargs.pop("edge_index"))
        else:
            inp = x
        x2 = self.dropout1(self.gnn(inp, *inputs, **kwargs))  # # (N, out_channels)
        if x.size(-1) == x2.size(-1):
            x = x + x2  # residual
            x = self.norm1(x)
        else:  # no residual if this layer is changing the model dim
            x = self.norm1(x2)
        x2 = self.linear2(self.dropout(torch.relu(self.linear1(x))))
        x = x + self.dropout2(x2)  # residual
        x = self.norm2(x)

        return x


class TransGNNLayer(TUF):

    def __init__(self, GNNClass, intermediate_fac=2, use_edge_type_embs=False, **layer_kwargs):
        init_args = inspect.getfullargspec(GNNClass.__init__)[0]
        needed_kwargs = {k: v for k, v in layer_kwargs.items() if k in init_args}

        size = get_config().hidden_size
        if GNNClass == GATConv:
            assert get_config().hidden_size % layer_kwargs["heads"] == 0
            out_size = get_config().hidden_size / layer_kwargs["heads"]
        else:
            out_size = size
        size = int(size)
        out_size = int(out_size)
        gnn = GNNClass(size, out_size, **needed_kwargs)

        if get_config().use_switch_gnn:
            gnn = SwitchGNN(gnn)
        if use_edge_type_embs:
            gnn = EdgeEmbeddings(gnn, size)

        linear1 = Linear(size, size * intermediate_fac)
        linear2 = Linear(size * intermediate_fac, size)

        super().__init__(size, gnn, linear1, linear2)


class SharedTransGNNLayer(TUF):
    """distinctly weighted GNN, shared linear layers"""
    def __init__(self, translayer: TransGNNLayer):
        new_gnn = copy.deepcopy(translayer.gnn)
        super().__init__(translayer.size, new_gnn, translayer.linear1, translayer.linear2)


class SimpleGNNLayer(nn.Module):
    def __init__(self, GNNClass, use_edge_type_embs=False, **layer_kwargs):
        super().__init__()

        h_size = get_config().hidden_size
        init_args = inspect.getfullargspec(GNNClass.__init__)[0]
        needed_kwargs = {k: v for k, v in layer_kwargs.items() if k in init_args}
        if GNNClass == GATConv:
            self.gnn = GNNClass(h_size, h_size//layer_kwargs["heads"], **needed_kwargs)
        else:
            self.gnn = GNNClass(h_size, h_size, **needed_kwargs)

        if get_config().use_switch_gnn:
            self.gnn = SwitchGNN(self.gnn)
        if use_edge_type_embs:
            self.gnn = EdgeEmbeddings(self.gnn, h_size)

        self.dropout1 = Dropout(get_config().dropout)

    def forward(self, x, *inputs, **kwargs):
        "x ~ (N, in_channels)"
        x = self.dropout1(torch.relu(self.gnn(x, *inputs, **kwargs)))  # # (N, out_channels)
        return x