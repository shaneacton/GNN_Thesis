import torch

from Code.Data.Graph.Embedders.relative_position_embedder import RelativePositionEmbedder
from Code.Data.Graph.Nodes.node_position import incompatible
from Code.Models.GNNs.LayerModules.layer_module import LayerModule
from Code.Training import device


class MessageModule(LayerModule):

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self._relative_positional_embedder = None

    def get_relative_positional_embedder(self, gcc, gec):
        if not self._relative_positional_embedder:
            self._relative_positional_embedder = RelativePositionEmbedder(self.channels, gcc, gec).to(device)
        return self._relative_positional_embedder

    def forward(self, edge_index_i, edge_index_j, x_i, x_j, size_i, encoding, **kwargs):
        """
        :param edge_index_i: (E)
        :param x_i: (E, in_channels)
        :param x_j: (E, in_channels)
        """
        return x_j

    def get_positional_embeddings(self, edge_index_i, edge_index_j, encoding) -> torch.Tensor:
        relative_positions = []
        if edge_index_i.size(0) != edge_index_j.size(0):
            raise Exception("e_id_i: " + repr(edge_index_i.size()) + " e_id_j: " + repr(edge_index_j.size()))
        for e in range(edge_index_i.size(0)):
            try:
                pos_i = encoding.graph.node_positions[edge_index_i[e]]
                pos_j = encoding.graph.node_positions[edge_index_j[e]]
            except Exception as ex:
                print("e_id_i: " + repr(edge_index_i.size()) + " e_id_j: " + repr(edge_index_j.size()))
                print("i[e]:", edge_index_i[e], "num node positions:", len(encoding.graph.node_positions), "e:", e)
                print("j[e]:", edge_index_j[e])
                print("i:",edge_index_i,"\n","j:",edge_index_j)
                raise ex
            if pos_i and pos_j:
                rel_pos = pos_j - pos_i
                relative_positions.append(rel_pos)
            else:
                # at least one of these nodes does not have a position eg: candidate node/ non span node
                relative_positions.append(incompatible)

        return self.get_relative_positional_embedder(encoding.graph.gcc, encoding.gec)(relative_positions)