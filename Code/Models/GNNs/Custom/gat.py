from typing import Tuple, Union, Optional

import torch
from torch import Tensor, nn
from torch_geometric.nn import GATConv
from torch_geometric.typing import OptTensor, OptPairTensor, Adj, Size


class Gat(GATConv):

    """
        wrapper around a regular GATConv to make the update rule asymmetric about sender/receiver
        also makes the GAT relational
    """

    def __init__(self, in_channels: Union[int, Tuple[int, int]], out_channels: int, num_edge_types=1, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        self.num_edge_types = num_edge_types
        self.lin = nn.Linear(2 * out_channels, out_channels)
        self.act = nn.ReLU()

        self._temp_edge_types: Tensor = None

        self.type_embeddings = nn.Embedding(num_edge_types, out_channels)
        self.type_map = nn.Linear(out_channels, out_channels)

        if num_edge_types > 1:
            layers = [nn.Linear(out_channels, out_channels) for _ in range(num_edge_types)]
            self.relational_transforms = nn.ModuleList(layers)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_types,
                size: Size = None, return_attention_weights=None):
        """cannot pass new args through the GAT, must bypass"""
        self._temp_edge_types = edge_types
        return super().forward(x, edge_index, size, return_attention_weights)

    def update(self, inputs: Tensor, x: Tensor) -> Tensor:
        """
            combines the agg messages with the original states to give assymetry and allow for preservation of
            node specific info such as positions
        """
        x_l, _ = x
        # print("custom gat got aggout:", inputs.size() , "x_l:", x_l.size())
        cat = torch.cat([inputs, x_l], dim=2)
        out = self.act(self.lin(cat))
        return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        """pretransforms the messages distinctly per edge type"""
        if self.num_edge_types > 1:
            x_j = self.relationally_transform(x_j)
        # type_embs = self.type_embeddings(self._temp_edge_types)
        # # print("types:", self._temp_edge_types.size())
        # x_j = x_j.squeeze_()
        # # print("x_j:", x_j.size(), "type_embs:", type_embs.size())
        # super_pos = x_j.add_(type_embs)
        #
        # x_j = self.act(self.type_map(super_pos))

        x_j = x_j.view(x_j.size(0), 1, x_j.size(-1))
        # print("x_j:", x_j.size())
        return super(Gat, self).message(x_j, alpha_j, alpha_i, index, ptr, size_i)

    def relationally_transform(self, x_j: Tensor):
        """
            breaks x_j up into same-edge-typed chunks, linearly transforms the chunks depending on type
            finally sews the transformed chunks back into the original tensor
            effectively transforms every message state differently depending on its edge type
        """
        x_j = x_j.squeeze_()
        # print("x_j:", x_j.size())
        # print("edge_types:", self._temp_edge_types.size(), self._temp_edge_types)
        if x_j.size(0) != self._temp_edge_types.size(0):
            raise Exception("Cannot do relational transforms. Given wrong number of edge types. Needed: " +
                            repr(x_j.size(0)) + " got: " + repr(self._temp_edge_types.size(0)) +
                            " make sure all self loops are accounted for in the edge info provided")
        # print("x_j:", x_j)
        for type in range(self.num_edge_types):
            indices = (self._temp_edge_types == type).nonzero().squeeze_()
            # print("type:", type, "idxs:", indices.size(), indices)
            features = x_j.index_select(dim=0, index=indices)
            # features = x_j[indices[0].item():indices[-1].item() + 1, :]
            # print("extracted features:", features.size(), x_j[indices[0].item():indices[-1].item() + 1, :].is_pinned())
            transformed_features = self.relational_transforms[type](features)
            # print("transformed feats:", transformed_features.size())
            # print("slice:", x_j[indices[0].item():indices[-1].item() + 1, :].size())
            x_j[indices[0].item():indices[-1].item() + 1, :] = transformed_features
            # for i, index in enumerate(indices):
            #     # print("index:", index)
            #     x_j[index.item(), :] = transformed_features[i, :]
            #     print("tfi:", transformed_features[i, :])
        return x_j.view(x_j.size(0), 1, x_j.size(-1))


