from torch import nn


class LayerModule(nn.Module):

    """represents either a message, update or aggregate components of a GNN layer"""

    def __init__(self):
        super().__init__()
