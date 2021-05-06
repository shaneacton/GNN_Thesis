from Code.HDE.Graph.graph import HDEGraph
from Code.Training.wikipoint import Wikipoint


class GraphBuilder:

    """
        a configurable utility to construct graph's wth various requirements
        first applies special edges, then applies structural edges, then fully connected edges
    """

    def __init__(self):
        pass

    def build(self, example: Wikipoint) -> HDEGraph:
        """
            1. Add nodes {Doc, Sent, Ent, Query, Cands}
            2. Add special connections {comention}.
                Special connections are named edge types which supersede structural connections
                eg: comention of ents overrides codocument etc
            3 Add structural connections {hierarchical }

            #todo add full hierarchical granularity to edges, ei: same doc, same sent, etc.
        """
        graph = HDEGraph(example)


    def encode(self, example: Wikipoint):
        """returns (node features, edge features)"""
