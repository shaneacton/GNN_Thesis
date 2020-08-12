import torch

from Code.Models.GNNs.LayerModules.relational_module import RelationalModule


class RelationalMessage(RelationalModule):

    def forward(self, x_j: torch.Tensor, edge_types: torch.Tensor):
        return super().forward(x_j, edge_types)