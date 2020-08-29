import torch

from Code.Models.GNNs.LayerModules.Message.message_module import MessageModule
from Code.Models.GNNs.LayerModules.relational_module import RelationalModule


class RelationalMessage(RelationalModule, MessageModule):
    
    def __init__(self, channels, num_bases, activation_type, dropout_ratio):
        RelationalModule.__init__(self, channels, channels, num_bases, activation_type, dropout_ratio)

    def forward(self, x_j: torch.Tensor, edge_types: torch.Tensor):
        return RelationalModule.forward(self, x_j, edge_types)