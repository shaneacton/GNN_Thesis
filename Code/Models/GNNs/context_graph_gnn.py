from torch import nn
from torch_geometric.nn import TopKPooling, SAGEConv

from Code.Data.Graph.context_graph import ContextGraph


class ContextGraphGNN(nn.Module):

    """
    takes in context graphs as inputs
    dynamically creates graph layers as needed based off edge types and state communications
    """

    """
    Plan: 
    get basic pipeline working with only current state updates, no special or previous/starting states
    add in support for previous/starting states - using state multiplexer
    move all encoder params into the CG-GNN with options to fine tune or not
    save/loading of cg-gnn with encoders
    """

    def __init__(self):
        super().__init__()
        base_layer_args = {"activation_type": nn.ReLU, "prop_type": SAGEConv, "pool_type": TopKPooling,
                    "pool_args": {"ratio": 0.8}}

    def forward(self, context: ContextGraph):
        # format of graph layers forward: (x, edge_index, batch, **kwargs)
        # get x, autodetect feature count
        # init layers based on detected in_features and init args