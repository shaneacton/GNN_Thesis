import torch

from Code.Models.GNNs.LayerModules.Prepare.prepare_module import PrepareModule
from Code.Models.GNNs.LayerModules.relational_module import RelationalModule


class RelationalPrep(RelationalModule, PrepareModule):

    def forward(self, x: torch.Tensor, node_types: torch.Tensor):
        # print("rel prep. x:", x.size(), "node_types", node_types.size())
        return RelationalModule.forward(self, x, node_types)
