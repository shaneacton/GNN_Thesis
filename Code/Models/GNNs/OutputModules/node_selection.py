from typing import List, Union

import torch
from torch import nn, Tensor

from Code.Data.Graph.context_graph import QAGraph
from Code.Models.GNNs.OutputModules.output_model import OutputModel
from Code.Training import device
from Code.constants import CONTEXT, QUERY


class NodeSelection(OutputModel):
    """predicts a probability for all/subset of the nodes"""

    def __init__(self, in_features):
        super().__init__(in_features)
        self.probability_mapper = nn.Linear(in_features, 1)
        self.softmax = nn.Softmax(dim=0)

    def get_output_from_graph_encoding(self, data, **kwargs):
        output_ids = self.get_output_ids_from_graph(data, **kwargs)
        batchwise_probabilities = self.get_probabilities(data.x, output_ids)
        return batchwise_probabilities

    def get_output_from_tensor(self, x: Tensor, **kwargs):
        if "output_ids" not in kwargs:
            raise Exception("must provide ids of output elements")

        output_ids = kwargs["output_ids"]
        return self.get_probabilities(x.squeeze(), output_ids)

    def get_typed_node_ids_from_graph(self, graph, node_type, source=None, **kwargs) -> List[int]:
        """return the node ids of each token"""
        token_nodes: List[int] = graph.typed_nodes[node_type]
        # print("getting", node_type, "nodes from", source)
        if source is None:
            source = [CONTEXT, QUERY]
        return self.get_nodes_from_source(graph, token_nodes, source)

    @staticmethod
    def get_nodes_from_source(graph: QAGraph, nodes: List, sources: Union[str, List[str]]):
        """
            the nodes selected from need to be faithful to the order of the tokens to match the answer spans
            returns a span ordered list of node ids, if multiple sources, these are ordered as provided
        """
        if isinstance(sources, str):
            sources = [sources]
        nodes = [graph.ordered_nodes[n] for n in nodes]
        ordered_nodes = []
        for source in sources:
            source_nodes = [n for n in nodes if n.source == source]
            source_nodes.sort()
            ordered_nodes.extend([graph.node_id_map[n] for n in source_nodes])
        return ordered_nodes

    def get_probabilities(self, vec, output_ids, node_offset=1):
        """
        :param output_ids: which elements in vec can be picked as an output
        todo: account for batching in node ids
        """
        if not isinstance(output_ids, List) or not isinstance(output_ids[0], List) \
            or not isinstance(output_ids[0][0], int):
            raise Exception("must provide output ids in list shaped [batch, batch_node_ids]")
        probs = []
        max_node_count = -1
        # must do final probability mapping separately due to differing classification node counts per batch item
        # print("output ids:", len(output_ids), "*", [len(ids) for ids in output_ids], output_ids)
        for graph_node_ids in output_ids:
            """
                for each node
            """
            graph_node_ids = [id + node_offset for id in graph_node_ids]
            if not isinstance(output_ids, torch.Tensor):
                node_ids = torch.tensor(graph_node_ids).to(device)
            else:
                node_ids = graph_node_ids
            # print("selecting node:", node_ids, "\nfrom", vec.size())
            choices = torch.index_select(vec, 0, node_ids)
            # print("choices:",choices.size())
            probabilities = self.probability_mapper(choices).view(-1)
            probabilities = self.softmax(probabilities)
            # print("single probs:", probabilities.size(), probabilities)
            probs.append(probabilities)
            max_node_count = max(max_node_count, len(graph_node_ids))

        for p in range(len(probs)):  # pad
            num_probs = probs[p].size(0)
            probs[p] = torch.cat([probs[p], torch.zeros(max_node_count - num_probs).to(device)])

        batchwise_probabilities = torch.stack(probs).view(len(probs), -1)
        # print("probs:", batchwise_probabilities.size(), batchwise_probabilities)
        return batchwise_probabilities

    def get_output_ids_from_graph(self, data, **kwargs):
        """
            returns 2d arrary shaped (batch, node_ids)
            here the number of node ids may vary between batch items
        """
        if isinstance(data.graphs, List):
            all_ids = []
            for g in data.graphs:
                ids = self.get_node_ids_from_graph(g, **kwargs)
                all_ids.append(ids)
            return all_ids
        else:
            return [self.get_node_ids_from_graph(data.graphs, **kwargs)]

    def get_node_ids_from_graph(self, graph, **kwargs):
        # override to make node selection on a subset only
        return list(range(len(graph.ordered_nodes)))  # all nodes
