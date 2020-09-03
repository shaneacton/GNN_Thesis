from typing import List

from torch import nn

from Code.Models.GNNs.LayerModules.Prepare.linear_prep import LinearPrep
from Code.Models.GNNs.LayerModules.Prepare.prepare_module import PrepareModule
from Code.Models.GNNs.LayerModules.Update.linear_update import LinearUpdate
from Code.Models.GNNs.Layers.graph_layer import GraphLayer
from Code.Training import device


class GraphTransformer(GraphLayer):

    """
        an implementation of the transformer layer using GNN components.
        uses the same patterns of normalisations, dropouts, activations, attention etc as the Transformer
    """

    def __init__(self, sizes: List[int], activation_type, dropout_ratio, heads, activation_kwargs=None):
        GraphLayer.__init__(self, sizes, None, None, None, activation_type=activation_type, dropout_ratio=dropout_ratio,
                            activation_kwargs=activation_kwargs)

        if self.input_size == self.output_size:
            # in == out can use trivial prep
            prep = PrepareModule(self.input_size, self.output_size, activation_type, dropout_ratio, activation_kwargs=activation_kwargs)
        else:
            # must map to the desired features
            prep = LinearPrep(self.input_size, self.output_size, 1, activation_type, dropout_ratio, activation_kwargs=activation_kwargs)

        prep.to(device)

        from Code.Models.GNNs.LayerModules.Message.attention_module import AttentionModule
        attention = AttentionModule(self.output_size, activation_type, dropout_ratio, activation_kwargs=activation_kwargs, heads=heads).to(device)
        update_module = LinearUpdate(self.output_size, 1, activation_type, dropout_ratio, activation_kwargs=activation_kwargs).to(device)

        self.multi_headed_attention = GraphLayer(sizes, [prep], [attention], [update_module])
        self.norm1 = nn.LayerNorm(self.output_size)

        self.linear1 = nn.Linear(self.output_size, self.output_size)
        self.linear2 = nn.Linear(self.output_size, self.output_size)

        self.norm2 = nn.LayerNorm(self.output_size)

    def forward(self, data):
        data, x = self.multi_headed_attention(data, return_after_prep=True)

        x = x + self.dropout(data.x, self.training)  # residual
        x = self.norm1(x)

        data.x = self.activate(self.linear1(x))
        data.x = self.linear2(self.dropout(data.x, self.training))
        x = x + self.dropout(data.x, self.training)

        data.x = self.norm2(x)

        return data



