from typing import Dict

from torch import Tensor

from Code.Data.Graph.Nodes.node import Node
from Code.Data.Text.Tokenisation.entity import Entity


class EntityNode(Node):

    """
    entry level node which contains only a current state vector
    """

    def __init__(self, entity:Entity):
        super().__init__()
        self.current_state: Tensor = entity.get_embedding()

    def get_node_states(self) -> Dict[str, Tensor]:
        return {"current_state": self.current_state}
