from Code.Models.GNNs.OutputModules.node_selection import NodeSelection


class CandidateSelection(NodeSelection):

    def get_node_ids_from_graph(self, graph, **kwargs):
        """return the node ids of each candidate"""
        return list(graph.candidate_nodes)
