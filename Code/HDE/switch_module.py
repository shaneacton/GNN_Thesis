import copy

from torch import nn, ModuleDict
from torch.nn import ModuleList


class SwitchModule(nn.Module):

    def __init__(self, module, num_types=None, types=None):
        super().__init__()
        modules = []
        if types is None:
            types = range(num_types)
        self.map = {}
        for i, t in enumerate(types):
            self.map[t] = i
            if i == 0:
                modules.append(module)
            else:  # makes a clone
                modules.append(copy.deepcopy(module))
        self.typed_modules = ModuleList(modules)

    def module(self, type):
        return self.typed_modules[self.map[type]]

    def forward(self, *inputs, type=None, **kwargs):
        return self.module(type)(*inputs, **kwargs)