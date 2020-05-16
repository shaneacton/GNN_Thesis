from typing import List, Set

import torch

from Code.Data.Graph.Edges.edge_relation import EdgeRelation
from Code.Data.Graph.Nodes.node import Node

from torch_geometric.data import Data

class ContextGraph:

    """
    Directed graph
    """

    def __init__(self):
        self.nodes: List[Node] = []
        self.edges: Set[EdgeRelation] = set()

    @property
    def edge_index(self):
        """
        converts edges into connection info for pytorch geometric
        """
        index = [[], []]
        for edge in self.edges:
            for i in range(2):
                index[i].append(edge[i])
                if not edge.directed:  # adds returning direction
                    index[i].append(edge[1-i])
        return index

    @property
    def data(self):
        """
        converts this graph into data ready to be fed into pytorch geometric
        """
        # todo should support nodes with different state types
        states_dict = {}
        for node in self.nodes:  # in order traversal
            for state_name in node.get_node_states().keys():
                if not state_name in states_dict:
                    states_dict[state_name] = []
                states_dict[state_name].append(node.get_node_states()[state_name])

        for state_name in states_dict.keys():
            states_dict[state_name] = torch.stack(states_dict[state_name], dim=0)
        return Data(edge_index=self.edge_index, **states_dict)

if __name__ == "__main__":
    x = torch.tensor([[2, 1], [5, 6], [3, 7], [12, 0]], dtype=torch.float)
    print(x.size())