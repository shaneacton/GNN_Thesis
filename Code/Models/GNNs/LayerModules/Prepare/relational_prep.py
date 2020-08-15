import torch

from Code.Models.GNNs.LayerModules.relational_module import RelationalModule


class RelationalPrep(RelationalModule):

    def forward(self, x: torch.Tensor, node_types: torch.Tensor):
        # print("rel prep. x:", x.size(), "node_types", node_types.size())
        return super().forward(x, node_types)