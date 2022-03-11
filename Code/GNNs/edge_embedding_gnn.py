import torch
from torch import nn
from torch_geometric.nn import GATConv

from Code.GNNs.wrap_gnn import WrapGNN
from Config.config import get_config


class EdgeEmbeddings(WrapGNN):

    def __init__(self, gnn_layer, hidden_size, num_edge_types, target_vectors=None):
        super().__init__(gnn_layer)
        self.target_vectors = ["x_j"]
        if hasattr(get_config(), "use_edge_types") and get_config().use_edge_types:
            self.embeddings = nn.Embedding(num_embeddings=num_edge_types, embedding_dim=hidden_size)

    def wrap_message(self, *args, **kwargs):
        if hasattr(get_config(), "use_edge_types") and get_config().use_edge_types:
            if "edge_types" in self.last_custom_kwargs:
                edge_types = self.last_custom_kwargs["edge_types"]
            else:
                edge_types = kwargs["edge_types"]

            edge_embs = self.embeddings(edge_types)
            for target in self.target_vectors:
                vec = kwargs[target]
                vec = self.add_type_embs(edge_embs, vec)
                kwargs[target] = vec
        return kwargs

    def add_type_embs(self, edge_embs, vec):
        dims = len(list(vec.size()))
        if dims > 2:
            """multiheaded attention module. add embs to all head channels"""
            if vec.size(1) * vec.size(2) == edge_embs.size(1):
                """no inflation, add unique emb to each head channel"""
                edge_embs = edge_embs.view(edge_embs.size(0), vec.size(1), -1)
                vec += edge_embs
            else:
                """inflation, add the same emb to each head channel"""
                vec += edge_embs.view(edge_embs.size(0), 1, -1)

        else:
            assert vec.size() == edge_embs.size(), "feature vec size: " + repr(vec.size()) + " does not match edge embs: " + repr(edge_embs.size())
            vec += edge_embs
        return vec


if __name__ == "__main__":
    hid = 6
    num_nodes = 5
    num_edge_types = 3

    base_gnn = GATConv(hid, hid, heads=3)
    edge_gnn = EdgeEmbeddings(base_gnn, hid, num_edge_types)

    x = torch.zeros(num_nodes, hid)
    print("nodes:", x.size())
    edge_index = torch.tensor([[0,1,2,3], [1,0,3,2]]).long()  # these are the edges we want added
    """here edges is an (E, 2) list defined as [[from_ids][to_ids]]"""
    print("edges:", edge_index.size())
    """
        A self edge connects a node to itself
        the GAT adds in self edges automatically. these new edges are always appended to the end of the edge index
        thus we need a special edge type for self edges. Here we use edgetype=2 for self edges
    """
    edge_types = torch.tensor([0, 0, 1, 1] + [2] * num_nodes).long()  # will initially mismatch in size compared to edge_index

    x_out = edge_gnn(x, edge_index, edge_types=edge_types, return_attention_weights=None)