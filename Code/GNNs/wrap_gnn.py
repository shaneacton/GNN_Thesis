import inspect

import torch
from torch import nn
from torch_geometric.nn import GATConv
import copy

from Code.GNNs.custom_gat import CustomGAT
from Config.config import conf


class WrapGNN(nn.Module):

    def __init__(self, gnn_layer):
        super().__init__()
        self.gnn_layer = gnn_layer
        self.backup_gnn_layer = copy.deepcopy(gnn_layer)
        for param in self.backup_gnn_layer.parameters():
            param.requires_grad = False

        # store originals
        self.base_forward = gnn_layer.forward
        self.base_message = gnn_layer.message
        self.forward_needed_args = inspect.getfullargspec(self.base_forward)[0]
        self.message_needed_args = inspect.getfullargspec(self.base_message)[0]

        # replace original with wrapper, so base layer will call out message func
        gnn_layer.message = self.message

        self.last_custom_kwargs = None
        # temporarily stores forward kwargs which cannot be passed through the gnn layers forward
        # these kwargs can then be accessed in the custom forward method

    def message(self, *args, **kwargs):
        """
            called by the message passing forward method.
            calls our wrap message, then calls the original base message
        """
        kwargs = self.wrap_message(*args, **kwargs)
        if self.base_message == self.message:
            """
                python saves bound messages as an object reference + method name. 
                Thus when saving and loading a wrap gnn, the original gnn's message method will be lost.
                We recover from this by keeping a clone of the layer, copying our trained weights into the clone, 
                and switching to the clone, which still retains its original message function
            """
            print("lost reference to base gnn message func. now has: " + repr(self.base_message) +
                            " gnn has: " + repr(self.gnn_layer.message), " backup has: " + repr(self.backup_gnn_layer.message))
            self.backup_gnn_layer.load_state_dict(self.gnn_layer.state_dict())  # load trained weights into backup
            self.gnn_layer = self.backup_gnn_layer  # retorre from backup
            self.backup_gnn_layer = copy.deepcopy(self.gnn_layer)  # create new backup
            # set trainable params
            for param in self.backup_gnn_layer.parameters():
                param.requires_grad = False
            for param in self.gnn_layer.parameters():
                param.requires_grad = True
            self.base_message= self.gnn_layer.message  # perform the wrap message switch again
            self.gnn_layer.message = self.message
            # recovery complete!

        return self.base_message(*args, **kwargs)

    def wrap_message(self, *args, **kwargs):
        """called before the base layers forward. used for pretransforms"""
        pass

    def forward(self, x, edge_index, *args, **kwargs):
        """caller triggers Wrap's forward, which then triggers the wrap_forward, and finally the base layers forward"""
        base_gnn_kwargs = {k: v for k, v in kwargs.items() if k in self.forward_needed_args}
        custom_kwargs = {k: v for k, v in kwargs.items() if k not in self.forward_needed_args}
        self.last_custom_kwargs = custom_kwargs
        self.wrap_forward(x, edge_index, *args, **kwargs)
        if "previous_attention_scores" in kwargs:
            base_gnn_kwargs.update({"previous_attention_scores": kwargs["previous_attention_scores"]})

        output = self.base_forward(x, edge_index, *args, **base_gnn_kwargs)
        self.last_custom_kwargs = None
        return output

    def wrap_forward(self, x, edge_index, *args, **kwargs):
        """called before the base layers forward. used for pretransforms"""
        pass


class EdgeEmbeddings(WrapGNN):

    def __init__(self, gnn_layer, hidden_size, num_edge_types, target_vectors=None):
        super().__init__(gnn_layer)
        if target_vectors is None:
            if isinstance(gnn_layer, CustomGAT):
                target_vectors = ["q_i"]
            else:
                target_vectors = ["x_j"]
        self.target_vectors = target_vectors
        self.embeddings = nn.Embedding(num_embeddings=num_edge_types, embedding_dim=hidden_size)

    def wrap_message(self, *args, **kwargs):
        edge_types = self.last_custom_kwargs["edge_types"]
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

