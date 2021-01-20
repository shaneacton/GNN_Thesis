from Code.HDE.graph import HDEGraph


class HDEEdge:

    def __init__(self, from_id, to_id, graph=None, type=None):
        self.graph: HDEGraph = graph
        self._type = type
        self.from_id = from_id
        self.to_id = to_id

        if type is None and graph is None:
            raise Exception("must provide type or graph to generate type from")

    def type(self):
        if self._type is None:
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