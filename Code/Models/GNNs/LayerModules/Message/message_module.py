import torch

from Code.Data.Graph.Embedders.relative_position_embedder import RelativePositionEmbedder
from Code.Models.GNNs.LayerModules.layer_module import LayerModule


class MessageModule(LayerModule):

    def __init__(self, channels, use_positional_encodings=False, gcc=None, gec=None):
        super().__init__()
        self.use_positional_encodings = use_positional_encodings
        self.channels = channels
        if self.use_positional_encodings:
            self.relative_positional_embedder = RelativePositionEmbedder(channels, gcc, gec)

    def forward(self, edge_index_i, edge_index_j, x_i, x_j, size_i, encoding, **kwargs):
        """
        :param edge_index_i: (E)
        :param x_i: (E, in_channels)
        :param x_j: (E, in_channels)
        """
        return x_j

    def get_positional_embeddings(self, edge_index_i, edge_index_j, encoding) -> torch.Tensor:
        relative_positions = []
        for e in range(edge_index_i.size(0)):
            pos_i = encoding.graph.node_positions[edge_index_i[e]]
            pos_j = encoding.graph.node_positions[edge_index_j[e]]
            rel_pos = pos_j - pos_i
            relative_positions.append(rel_pos)

        return self.relative_positional_embedder(relative_positions)