import torch
from torch_geometric.nn.inits import glorot, uniform

from Code.Models.GNNs.LayerModules.layer_module import LayerModule
from torch.nn import Parameter

from Code.Training import device


class RelationalModule(LayerModule):
    """
    typically either a preparation or message module
    switches weight function based on edge/node types

    allows different nodes and edges to use distinct learned functions
    """

    def __init__(self, in_channels, out_channels, num_bases):
        LayerModule.__init__(self)
        self.num_bases = num_bases
        self.out_channels = out_channels
        self.in_channels = in_channels

        self.coefficients = None  # ~ (num_relations, num_bases)
        self.basis = Parameter(torch.Tensor(num_bases, in_channels, out_channels))

        self.bias = Parameter(torch.Tensor(out_channels))

        self.max_type_id = -1

        self.reset_parameters()


    def reset_parameters(self):
        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.bias)
        if self.coefficients is not None:
            uniform(size, self.coefficients)

    def handle_types(self, types: torch.Tensor):
        """
        if this forward pass has encountered a new type not previously seen
        the coefficients matrix must be updated to include these new types
        """
        max_id, _ = torch.max(types, dim=0)
        max_id = max_id.item()

        num_new_types = max_id - self.max_type_id
        if num_new_types <= 0:
            return
        new_weights = torch.Tensor(num_new_types, self.num_bases).to(device)
        uniform(self.num_bases * self.in_channels, new_weights)
        if self.coefficients is not None:  # must concat these new weights onto the old weights
            old_weights = self.coefficients.data
            new_weights = torch.cat([old_weights, new_weights], dim=0)
        # print("adding new coefficients", num_new_types, "max:", max_id, "types", types)
        self.coefficients = Parameter(new_weights).to(device)
        self.register_parameter("Coefficients", self.coefficients)

        self.max_type_id = max_id

    @property
    def num_relations(self):
        return self.max_type_id + 1

    def get_relational_weights(self, types: torch.Tensor) -> torch.Tensor:
        """
        Projects each type to a point in the basis space
        thus each type gets a unique transformation using only num_bases transformation primitives

        :returns (E/N, i, o) vector which has switched the transformation in dim0 based on node/edge types
        """

        self.handle_types(types)
        print("coeffs:", self.coefficients)
        # print("basis:", self.basis)
        rel_w = torch.matmul(self.coefficients, self.basis.view(self.num_bases, -1))
        # after matmul, each col in rel_w is a linear combination of the basis vectors
        rel_w = rel_w.view(self.num_relations, self.in_channels, self.out_channels)
        rel_w = torch.index_select(rel_w, 0, types)  # (E/N, i, o)
        return rel_w

    def forward(self, x: torch.Tensor, types: torch.Tensor):
        """
        :param x: either x ~ (N,f) for prep or x_j ~ (E,f) for message
        :param types: node or edge type ids
        """
        rel_w = self.get_relational_weights(types)

        print("w:",rel_w.size(), "x:", x.size(), "x unsqueezed:", x.unsqueeze(1).size())
        print("rel_w:",rel_w)
        # general bmm ~ (b,n,m) * (b,m,p) = (b,n,p)
        # (E/N, 1, f) * (E/N, i, o) = (E/N, 1, o)
        out = torch.bmm(x.unsqueeze(1), rel_w).squeeze(-2)  # (E/N, o)
        return out
