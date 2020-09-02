from typing import List

from torch import nn

from Code.Data.Graph.Embedders.graph_encoding import GraphEncoding
from Code.Models.GNNs.LayerModules.Message.attention_module import AttentionModule
from Code.Models.GNNs.LayerModules.Prepare.linear_prep import LinearPrep
from Code.Models.GNNs.LayerModules.Prepare.prepare_module import PrepareModule
from Code.Models.GNNs.LayerModules.Update.linear_update import LinearUpdate
from Code.Models.GNNs.Layers.graph_layer import GraphLayer


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
        attention = AttentionModule(self.output_size, activation_type, dropout_ratio, activation_kwargs=activation_kwargs)
        update = LinearUpdate(self.output_size, 1, activation_type, dropout_ratio, activation_kwargs=activation_kwargs, heads=heads)

        self.multi_headed_attention = GraphLayer(sizes, [prep], [attention], [update])
        self.norm1 = nn.LayerNorm(self.output_size)

        self.linear1 = nn.Linear(self.output_size, self.output_size)
        self.linear2 = nn.Linear(self.output_size, self.output_size)

        self.norm2 = nn.LayerNorm(self.output_size)

    def forward(self, data: GraphEncoding) -> GraphEncoding:
        x = data.x
        data = self.multi_headed_attention(data)

        x = x + self.dropout(data.x)  # residual
        x = self.norm1(x)

        data.x = self.linear2(self.dropout(self.activate(self.linear1(x))))
        x = x + self.dropout(data.x)

        data.x = self.norm2(x)

        return data



