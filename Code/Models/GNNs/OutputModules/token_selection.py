from Code.Data.Graph.Nodes.token_node import TokenNode
from Code.Models.GNNs.OutputModules.node_selection import NodeSelection


class TokenSelection(NodeSelection):
    def get_node_ids_from_graph(self, graph):
        """return the node ids of each token"""
        return list(graph.typed_nodes[TokenNode])