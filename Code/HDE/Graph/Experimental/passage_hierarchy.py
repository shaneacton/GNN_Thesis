from typing import Dict, List

from Code.HDE.Graph.node import HDENode
from Code.constants import TOKEN, ENTITY, SENTENCE, PASSAGE


class PassageHierarchy:
    """
        Our system assumes that a passage is the largest granularity of text that can be grouped together
        Ie there is no support for documents containing multiple passages. Or subsets of documents which are related

        This system assumes a set of free floating passages.
    """

    hierarchy_order = [TOKEN, ENTITY, SENTENCE, PASSAGE]

    def __init__(self):
        self.level_nodes: Dict[str, List[HDENode]] = {}

        # maps each node's in_layer_id to all its childrens in_layer_id's

        self.level_childrens_map: Dict[str, Dict[int, List[int]]] = {}

    def add_node(self, node: HDENode, level: str, parent_id_in_layer: int, parent_level: str):
        if level not in self.level_nodes:
            self.add_level(level)
        level = self.level_nodes[level]
        id_in_level = len(level)
        level.append(node)

        if parent_id_in_layer not in self.level_childrens_map[parent_level].keys():
            self.level_childrens_map[parent_level][parent_id_in_layer] = []
        self.level_childrens_map[parent_level][parent_id_in_layer].append(id_in_level)
        return id_in_level

    def add_level(self, level):
        self.level_nodes[level] = []
        self.level_childrens_map[level] = {}

    def connect_interlevel(self, lowest_level_num, highest_level_num, connect_intermediate=False):
        """connecting intermediately will be more efficient than connecting one at a time"""
        highest_level = self.levels[highest_level_num]
        highest_layer_name = self.names_map[highest_level_num]
        """if the level diff is more than 1, then we must recursively collect all our nth degree children"""

        parent_level_num = highest_level_num
        child_level_num = highest_level_num - 1

        high_parents_first_degree_children = {p: self.level_childrens_map[parent_level_num][p] for p in range(len(highest_level))}
        high_parents_nth_children = high_parents_first_degree_children

        while child_level_num != lowest_level_num:
            """for each recursive step, gather grandchildren, set them to  children, repeat"""
            high_parents_grandchildren = {p: [] for p in range(len(highest_level))}
            child_layer_name = self.names_map[child_level_num]
            for high_p in range(len(highest_level)):  # for each original parent
                for child_id in high_parents_nth_children[high_p]:  # for each of their nth degree children in this layer
                    childs_children: List[int] = self.level_childrens_map[child_level_num][child_id]
                    high_parents_grandchildren[high_p].extend(childs_children)

                if connect_intermediate:  # connects parents to their children, not grandchildren
                    """these should be connected to the parent nodes"""
                    _type = highest_layer_name + "_contains_" + child_layer_name
                    child_ids_in_graph = [self.levels[child_level_num][c].id_in_graph for c in high_parents_nth_children[high_p]]
                    connect_one_to_all(highest_level[high_p].id_in_graph, child_ids_in_graph, self.graph, _type)

                """swap children for grandchildren to be expanded again in the next it"""
                high_parents_nth_children[high_p] = high_parents_grandchildren[high_p]

            child_level_num -= 1

        assert child_level_num == lowest_level_num

        """by now, our nth degree children are in our target lowest_level"""
        for high_p in range(len(highest_level)):  # for each original parent
            children_ids_in_level = high_parents_nth_children[high_p]
            ids_in_graph = [self.levels[child_level_num][c].id_in_graph for c in children_ids_in_level]
            _type = highest_layer_name + "_contains_" + self.names_map[child_level_num]
            connect_one_to_all(highest_level[high_p].id_in_graph, ids_in_graph, self.graph, _type)

    def get_all_node_ids(self):
        """returns all nodes under this passage, at all levels"""
        all_ids = []
        for lev, nodes in self.level_nodes.items():
            ids = [n.id_in_graph for n in nodes]
            all_ids.extend(ids)
        return all_ids
