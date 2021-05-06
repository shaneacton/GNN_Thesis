from typing import List, Dict

from Code.HDE.Graph.graph import HDEGraph
from Code.HDE.Graph.node import HDENode
from Code.Utils.graph_utils import fully_connect, window_connect, connect_one_to_all, connect_all_to_all


class NodeHierarchy:

    """
        A temporary data structure to help build context graphs from wikihop examples
        each level of the hierarchy is an ordered set of nodes.

        does not
    """

    def __init__(self, graph: HDEGraph):
        self.graph = graph
        self.levels: List[List[HDENode]] = []

        self.names_map = {0: "word", 1: "sentence", 2: "paragraph", 3: "sub_query", 4: "query"}
        self.id_to_name_map = {v: k for k, v in self.names_map.items()}

        # maps each node's in_layer_id to its parents in_layer_id
        self.level_parents_map: List[Dict[int, int]] = [{} for _ in self.names_map]

        # maps each node's in_layer_id to all its childrens in_layer_id's
        self.level_childrens_map: List[Dict[int, List[int]]] = [{} for _ in self.names_map]

    def get_all_active_levels(self):
        return [0, 1, 2]

    def add_node(self, node: HDENode, level_num: int, parent_id_in_layer: int):
        level = self.levels[level_num]
        id_in_level = len(level)
        level.append(node)
        if parent_id_in_layer is not None:  # none indicates a top level node
            self.level_parents_map[level_num][id_in_level] = parent_id_in_layer

        parent_level = level_num + 1
        if parent_id_in_layer not in self.level_childrens_map[parent_level].keys():
            self.level_childrens_map[parent_level][parent_id_in_layer] = []
        self.level_childrens_map[parent_level][parent_id_in_layer].append(id_in_level)
        return id_in_level

    def connect_intralevel(self, level_num, strategy:str, connect_between_paragraphs=False):
        """
            Connects nodes at the same level, with a configurable strategy.

            strategy could be fully_connect, window_x, sequential
        """
        level: List[HDENode] = self.levels[level_num]
        type_name = self.names_map[level_num] + "_" + strategy
        paragraph_level_num = self.id_to_name_map["paragraph"]
        node_groups = {1: level}
        if level_num < paragraph_level_num:
            """
                sub paragraph nodes are connected differently than super paragraph nodes.
                here, sub paragraph nodes should only be connected if they share a paragraph
            """
            paragraph_nodes = {p: [] for p in range(len(self.levels[paragraph_level_num]))}
            for i, node in enumerate(level):
                paragraph_id = self.get_parent_at_level(level_num, i, paragraph_level_num)
                paragraph_nodes[paragraph_id].append(node)
            node_groups = paragraph_nodes
        group_ids_in_graph = {}
        for g, group in enumerate(node_groups):  # intragroup
            ids = [node.id_in_graph for node in group]
            group_ids_in_graph[g] = ids
            if strategy == "fully_connect":
                fully_connect(ids, self.graph, type_name)
            if strategy == "window_x" or strategy == "sequential":
                if strategy == "sequential":  # sequential is just a window connection with WS=1
                    window_size = 1
                else:
                    window_size = int(strategy.split("_")[1])
                window_connect(ids, window_size, self.graph, type_name, wrap=True)

        if connect_between_paragraphs and strategy == "fully_connect" and len(node_groups) > 1:  # intergroup
            type_name += "compliment"
            for g1, group1 in enumerate(node_groups):  # intragroup
                for g2, group2 in enumerate(node_groups):  # intragroup
                    if g1 == g2:
                        pass
                    connect_all_to_all(group_ids_in_graph[g1], group_ids_in_graph[g1], self.graph, type_name)

    def get_parent_at_level(self, child_level_num, child_id_in_level, parent_level_num):
        assert child_level_num < parent_level_num
        level_diff = parent_level_num - child_level_num  # how many levels to  climb up
        for i in range(level_diff):
            sub_level_num = child_level_num + i  # starts off at childs level and climbs up
            parent_id = self.level_parents_map[sub_level_num][child_id_in_level]
            child_id_in_level = parent_id
        return parent_id

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



