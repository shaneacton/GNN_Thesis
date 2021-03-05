import copy

from torch import nn, ModuleDict
from torch.nn import ModuleList

GLOBAL = "global"


class SwitchModule(nn.Module):
    """makes a clone of the given layer or each type"""

    def __init__(self, module, types=None, include_global=False):
        super().__init__()
        modules = []
        self.map = {}
        if include_global:
            types.append(GLOBAL)
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