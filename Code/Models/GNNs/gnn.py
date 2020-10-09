from typing import List

from torch import nn

from Code.Models.GNNs.gnn_component import GNNComponent


class GNN(GNNComponent, nn.Module):

    def __init__(self, sizes: List[int], activation_type, dropout_ratio, activation_kwargs=None):
        nn.Module.__init__(self)
        GNNComponent.__init__(self, sizes, activation_type, dropout_ratio, activation_kwargs=activation_kwargs)




