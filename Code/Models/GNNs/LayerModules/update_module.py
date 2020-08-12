import copy

from torch import nn
from torch_geometric.nn import RGCNConv

from Code.Models.GNNs.LayerModules.layer_module import LayerModule
from Code.Training import device


class UpdateModule(LayerModule):

    def __init__(self):
        super().__init__()

    def forward(self, aggr_out, x, **kwargs):
        """

        :param aggr_out:
        :param x:
        :param kwargs:
        :return:
        """
        return aggr_out


class Test(nn.Module):

    def __init__(self):
        super().__init__()
        self.base = nn.Linear(1,2).to(device)
        self.copy_list = []
        self.copies = None

    def add_copy(self):
        self.copy_list.append(copy.deepcopy(self.base))
        self.copies = nn.ModuleList(self.copy_list)


if __name__ == "__main__":
    test = Test()
    test.add_copy()
    print(test)
    rgcn = RGCNConv()