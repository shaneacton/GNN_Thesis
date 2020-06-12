from abc import ABC, abstractmethod


class State(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def get_named_state_tensors(self):
        raise NotImplementedError()
