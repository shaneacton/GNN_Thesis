from abc import ABC, abstractmethod


class State(ABC):

    @abstractmethod
    def get_state_tensors(self):
        raise NotImplementedError()
