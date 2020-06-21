from abc import ABC, abstractmethod
from typing import Dict

from torch import Tensor


class State(ABC):

    GLOBAL = "global"
    NODE = "node"
    EDGE = "edge"

    def __init__(self, name, type=NODE):
        self.type = type
        self.name = name

    @abstractmethod
    def get_named_state_tensors(self) -> Dict[str, Tensor]:
        raise NotImplementedError()


