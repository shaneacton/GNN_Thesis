from typing import List

from Code.HDE.Graph.Experimental.passage_hierarchy import PassageHierarchy


class ContextHierarchy:

    """
        A collection of passage hierarchies. Represents the full context of a given example
    """

    def __init__(self):
        self.passages: List[PassageHierarchy] = []

    def get_all_node_ids(self):
        ids = []
        for p in self.passages:
            ids.extend(p.get_all_node_ids())
        return ids

    def fully_connect_layer(self, layer):
        """
            fully connects the nodes at this level in every passage, both inter and intra passage

        """