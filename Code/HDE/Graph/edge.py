from Code.HDE.Graph.graph import HDEGraph
from Code.constants import SEQUENTIAL


class HDEEdge:

    def __init__(self, from_id, to_id, graph=None, type=None, safe_mode=False):
        self.graph: HDEGraph = graph
        self._type = type
        self.from_id = from_id
        self.to_id = to_id

        if from_id == to_id and not safe_mode:
            raise Exception("self loops are optionally added in by the gnn. not modeled by hdegraph")

        if type is None and graph is None and not safe_mode:
            raise Exception("must provide type or graph to generate type from")

    def is_bidirectional(self):
        # edges which are x2y are bidirectional. Generic edges like Comention or Entity are unidirectional
        nodes = self.graph.ordered_nodes
        f, t = nodes[self.from_id], nodes[self.to_id]
        types = sorted([f.type, t.type])
        return self._type is None and types[0] != types[1]

    def type(self, reverse=False):
        if self._type is None:  # default if no type is provided
            nodes = self.graph.ordered_nodes
            f, t = nodes[self.from_id], nodes[self.to_id]
            types = sorted([f.type, t.type])
            if reverse and self.is_bidirectional():
                return types[1] + "2" + types[0]
            else:
                return types[0] + "2" + types[1]
        if reverse and self._type == SEQUENTIAL:
            return self._type + "_reverse"
        return self._type

    def __eq__(self, other):
        """same edge if it connects the same two nodes"""
        s = sorted([self.from_id, self.to_id])
        o = sorted([other.from_id, other.to_id])
        return s[0] == o[0] and s[1] == o[1]

    @property
    def edge_hash(self):
        return tuple(sorted([self.from_id, self.to_id]))

    def __hash__(self):
        """same edge if it connects the same two nodes"""
        ids = self.edge_hash
        return hash(ids[0]) * 7 + hash(ids[1]) * 13

    def __repr__(self) -> str:
        return "Edge: " + self.type()


if __name__ == "__main__":

    e1 = HDEEdge(0, 1, type="whatever")
    e2 = HDEEdge(1, 0, type="whatever")
    print("hashes:", e1.__hash__(), e2.__hash__())
    print("eq:", e1 == e2)
    edges = {e1}

    print(e2 in edges)