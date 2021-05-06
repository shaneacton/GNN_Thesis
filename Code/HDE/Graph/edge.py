from Code.HDE.Graph.graph import HDEGraph


class HDEEdge:

    def __init__(self, from_id, to_id, graph=None, type=None):
        self.graph: HDEGraph = graph
        self._type = type
        self.from_id = from_id
        self.to_id = to_id

        if from_id == to_id:
            raise Exception("self loops are optionally added in by the gnn. not modeled by hdegraph")

        if type is None and graph is None:
            raise Exception("must provide type or graph to generate type from")

    def type(self):
        if self._type is None:  # default if no type is provided
            nodes = self.graph.ordered_nodes
            f, t = nodes[self.from_id], nodes[self.to_id]
            types = sorted([f.type, t.type])
            return types[0] + "2" + types[1]

        return self._type

    def __eq__(self, other):
        """same edge if it connects the same two nodes"""
        s = sorted([self.from_id, self.to_id])
        o = sorted([other.from_id, other.to_id])
        return s[0] == o[0] and s[1] == o[1]

    def __hash__(self):
        """same edge if it connects the same two nodes"""
        ids = sorted([self.from_id, self.to_id])

        return hash(ids[0]) * 7 + hash(ids[1]) * 13


if __name__ == "__main__":

    e1 = HDEEdge(0, 1, type="whatever")
    e2 = HDEEdge(1, 0, type="whatever")
    print("hashes:", e1.__hash__(), e2.__hash__())
    print("eq:", e1 == e2)
    edges = {e1}

    print(e2 in edges)