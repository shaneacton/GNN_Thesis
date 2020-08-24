from Code.Models.GNNs.OutputModules.node_selection import NodeSelection


class CandidateSelection(NodeSelection):

    def get_node_ids(self, data):
        """return the node ids of each candidate"""
        return data.graph.candidate_nodes
