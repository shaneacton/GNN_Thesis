
class GraphBuilder:

    def __init__(self):
        pass

    def build(self):
        """
            1. Add nodes {Doc, Sent, Ent, Query, Cands}
            2. Add special connections {comention}.
                Special connections are named edge types which supersede structural connections
                eg: comention of ents overrides codocument etc
            3 Add structural connections {hierarchical }

            #todo add full hierarchical granularity to edges, ei: same doc, same sent, etc.
        """

    def encode(self):
        """returns node features"""