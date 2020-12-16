from Code.Models.GNNs.OutputModules.node_selection import NodeSelection
from Code.Models.loss_funcs import get_span_element_loss


class CandidateSelection(NodeSelection):

    def get_node_ids_from_graph(self, graph, **kwargs):
        """return the node ids of each candidate"""
        # print("using cand selection. nodes:", graph.candidate_nodes)
        return sorted(list(graph.candidate_nodes))

    def get_output_from_graph_encoding(self, data, **kwargs):
        probs = super().get_output_from_graph_encoding(data, **kwargs)
        if 'answer' in kwargs:
            loss = get_span_element_loss(kwargs["answer"], probs)
            return loss, probs
        return probs

    def get_probabilities(self, vec, output_ids):
        return super(CandidateSelection, self).get_probabilities(vec, output_ids, node_offset=0)