import math

import torch
from torch import nn, tensor, Tensor
from torch.nn import Parameter, init


class RelationalLinear(nn.Module):

    def __init__(self, in_channels, out_channels, num_edge_types):
        super().__init__()
        self.num_edge_types = num_edge_types
        self.rel_weights = [Parameter(Tensor(in_channels, out_channels)) for _ in range(num_edge_types)]
        # self.rel_biases = [Parameter(Tensor(out_channels)) for _ in range(num_edge_types)]

        self.register_parameters()
        self.reset_parameters()

    def forward(self, x_j: Tensor, edge_types: Tensor):
        print("types:", [type.item() for type in edge_types])
        effective_weight = torch.stack([self.rel_weights[type.item()] for type in edge_types])
        # effective_bias = torch.stack([self.rel_biases[type] for type in edge_types])
        x_j = x_j.matmul(effective_weight)
        x_j = torch.stack([x_j[i, idx, :] for i, idx in enumerate(edge_types)])
        # x_j += effective_bias
        return x_j

    def reset_parameters(self) -> None:
        for i, rel_weight in enumerate(self.rel_weights):
            init.kaiming_uniform_(rel_weight, a=math.sqrt(5))
            # fan_in, _ = init._calculate_fan_in_and_fan_out(rel_weight)
            # bound = 1 / math.sqrt(fan_in)
            # init.uniform_(self.rel_biases[i], -bound, bound)

    def register_parameters(self):
        for i, rel_weight in enumerate(self.rel_weights):
            self.register_parameter("rel_weight_" + repr(i), rel_weight)
            # self.register_parameter("rel_bias_" + repr(i), self.rel_biases[i])


if __name__ == "__main__":

    x = tensor([[1.,2.,3.,4.], [5.,6.,7.,8.]])
    # x = tensor([5.,6.,7.,8.])

    print("x:", x.size(), x)

    weight = Parameter(Tensor(4, 3))
    weight2 = Parameter(Tensor(4, 3))
    idxs = [0, 1]


    # general bmm ~ (b,n,m) * (b,m,p) = (b,n,p)
    # want in(b,f) * (2,) = (b, f')

    torch.manual_seed(7)
    # init.kaiming_uniform_(weight, a=math.sqrt(5))
    # init.kaiming_uniform_(weight2, a=math.sqrt(5))
    # weight_comb = torch.stack([weight.data, weight2.data])
    # # print("w:", weight_comb.size(), weight_comb)
    # y = x.matmul(weight_comb)
    # y = torch.stack([y[i, idx, :] for i, idx in enumerate(idxs)])

    rel = RelationalLinear(4, 3, 2)
    y = rel(x, tensor(idxs))


    print("y:", y.size(), y)


    # y: torch.Size([2, 3]) tensor([[ 1.0104,  1.1239, -0.2205],
    #         [ 2.1659,  0.7370, -0.2181]], grad_fn=<MmBackward>)


    # y: torch.Size([2, 3]) tensor([[ 6.5571e-01, -9.0325e-04,  1.1769e+00],
    #         [ 7.9066e-01, -9.6309e-01,  1.6197e+00]], grad_fn=<MmBackward>)