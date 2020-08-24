import torch

from Code.Models.GNNs.LayerModules.Message.message_module import MessageModule
from Code.Models.GNNs.LayerModules.relational_module import RelationalModule


class RelationalMessage(RelationalModule, MessageModule):
    
    def __init__(self, channels, num_bases):
        super().__init__(channels, channels, num_bases)

    def forward(self, x_j: torch.Tensor, edge_types: torch.Tensor):
        return RelationalModule.forward(self, x_j, edge_types)