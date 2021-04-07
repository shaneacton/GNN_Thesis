import inspect

import torch
from torch import nn
from torch_geometric.nn import GATConv


class WrapGNN(nn.Module):

    def __init__(self, gnn_layer):
        super().__init__()
        self.gnn_layer = gnn_layer

        # store originals
        self.base_forward = gnn_layer.forward
        self.base_message = gnn_layer.message

        # replace originals with wrappers
        gnn_layer.forward = self.forward
        gnn_layer.message = self.message

        self.last_custom_kwargs = None
        # temporarily stores forward kwargs which cannot be passed through the gnn layers forward
        # these kwargs can then be accessed in the custom forward method

    def message(self, x_j, index, *args, **kwargs):
        """called by the message passing forward method"""
        x_j = self.wrap_message(x_j, index, *args, **kwargs)
        return self.base_message(x_j=x_j, index=index, *args, **kwargs)

    def wrap_message(self,  x_j, index, *args, **kwargs):
        return x_j

    def forward(self, x, edge_index, *args, **kwargs):
        needed = inspect.getfullargspec(self.base_forward)[0]
        base_gnn_kwargs = {k: v for k, v in kwargs.items() if k in needed}
        custom_kwargs = {k: v for k, v in kwargs.items() if k not in needed}
        self.last_custom_kwargs = custom_kwargs
        self.wrap_forward(x, edge_index, *args, **kwargs)
        output = self.base_forward(x, edge_index, *args, **base_gnn_kwargs)
        self.last_custom_kwargs = None
        return output

    def wrap_forward(self, x, edge_index, *args, **kwargs):
        # print("base wrap forward")
        pass


class EdgeEmbeddings(WrapGNN):

    def __init__(self, gnn_layer, hidden_size, num_edge_types):
        super().__init__(gnn_layer)
        self.embeddings = nn.Embedding(num_embeddings=num_edge_types, embedding_dim=hidden_size)

    def wrap_message(self, x_j, index, *args, **kwargs):
        edge_types = self.last_custom_kwargs["edge_types"]
        edge_embs = self.embeddings(edge_types)
        dims = len(list(x_j.size()))
        if dims > 2:
            """multiheaded attention module. add embs to all head channels"""
            if x_j.size(1) * x_j.size(2) == edge_embs.size(1):
                """no inflation, add unique emb to each head channel"""
                edge_embs = edge_embs.view(edge_embs.size(0), x_j.size(1), -1)
                x_j += edge_embs
            else:
                """inflation, add the same emb to each head channel"""
                x_j += edge_embs.view(edge_embs.size(0), 1, -1)

        else:
            x_j += edge_embs
        return x_j


if __name__ == "__main__":
    hid = 6
    num_nodes = 5
    num_edge_types = 3

    base_gnn = GATConv(hid, hid, heads=3)
    edge_gnn = EdgeEmbeddings(base_gnn, hid, num_edge_types)

    x = torch.zeros(num_nodes, hid)
    edge = torch.tensor([[0,1,2,3], [1,0,3,2]]).long()
    edge_types = torch.tensor([0, 0, 1, 1] + [2] * num_nodes).long()

    x_out = edge_gnn(x, edge, edge_types=edge_types, return_attention_weights=None)

