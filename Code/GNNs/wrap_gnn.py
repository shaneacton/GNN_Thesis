import inspect

import torch
from torch import nn
from torch_geometric.nn import GATConv
import copy

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

        print("wrap init called. base message:", self.base_message, "gnn message:", gnn_layer.message)

    def message(self, x_j, index, *args, **kwargs):
        """
            called by the message passing forward method.
            calls our wrap message, then calls the original base message
        """
        x_j = self.wrap_message(x_j, index, *args, **kwargs)
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

        return self.base_message(x_j=x_j, index=index, *args, **kwargs)

    def wrap_message(self,  x_j, index, *args, **kwargs):
        """called before the base layers forward. used for pretransforms"""
        return x_j

    def forward(self, x, edge_index, *args, **kwargs):
        """caller triggers Wrap's forward, which then triggers the wrap_forward, and finally the base layers forward"""
        base_gnn_kwargs = {k: v for k, v in kwargs.items() if k in self.forward_needed_args}
        custom_kwargs = {k: v for k, v in kwargs.items() if k not in self.forward_needed_args}
        self.last_custom_kwargs = custom_kwargs
        self.wrap_forward(x, edge_index, *args, **kwargs)
        output = self.base_forward(x, edge_index, *args, **base_gnn_kwargs)
        self.last_custom_kwargs = None
        return output

    def wrap_forward(self, x, edge_index, *args, **kwargs):
        """called before the base layers forward. used for pretransforms"""
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

