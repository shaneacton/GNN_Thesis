from typing import List

from torch import nn

from Code.Models.GNNs.gnn_component import GNNComponent


class LayerModule(GNNComponent, nn.Module):

    """represents either a message, update or aggregate components of a GNN layer"""

    def __init__(self, sizes: List[int], activation_type, dropout_ratio, activation_kwargs=None):
        """
        creates activation, overriding modules responsibility to use it
        """
        nn.Module.__init__(self)
        GNNComponent.__init__(self, sizes, activation_type, dropout_ratio, activation_kwargs=activation_kwargs)


