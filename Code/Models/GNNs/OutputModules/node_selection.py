from typing import List

import torch
from torch import nn, Tensor

from Code.Data.Graph.Embedders.graph_encoding import GraphEncoding
from Code.Models.GNNs.OutputModules.output_model import OutputModel
from Code.Training import device


class NodeSelection(OutputModel):
    """predicts a probability for all/subset of the nodes"""

    def __init__(self, in_features):
        super().__init__(in_features)
        self.probability_mapper = nn.Linear(in_features, 1)
        self.softmax = nn.Softmax(dim=0)

    def get_output_from_graph_encoding(self, data: GraphEncoding, **kwargs):
        output_ids = self.get_output_ids_from_graph(data)
        batchwise_probabilities = self.get_probabilities(data.x, output_ids)
        return batchwise_probabilities

    def get_output_from_tensor(self, x: Tensor, **kwargs):
        if "output_ids" not in kwargs:
            raise Exception("must provide ids of output elements")

        output_ids = kwargs["output_ids"]
        return self.get_probabilities(x.squeeze(), output_ids)

    def get_probabilities(self, vec, output_ids):
        """
        :param output_ids: which elements in vec can be picked as an output
        """
        probs = []
        max_node_count = -1
        # must do final probability mapping separately due to differing classification node counts per batch item
        for graph_node_ids in output_ids:
            if not isinstance(output_ids, torch.Tensor):
                node_ids = torch.tensor(graph_node_ids).to(device)
            else:
                node_ids = graph_node_ids
            # print("selecting node:", node_ids, "\nfrom", vec.size())
            choices = torch.index_select(vec, 0, node_ids)
            # print("choices:",choices.size())
            probabilities = self.probability_mapper(choices).view(-1)
            probabilities = self.softmax(probabilities)
            # print("single probs:", probabilities)
            probs.append(probabilities)
            max_node_count = max(max_node_count, len(graph_node_ids))

        for p in range(len(probs)):
            num_probs = probs[p].size(0)
            probs[p] = torch.cat([probs[p], torch.zeros(max_node_count - num_probs)])  # pad

        batchwise_probabilities = torch.stack(probs).view(len(probs), -1)
        return batchwise_probabilities

    def get_output_ids_from_graph(self, data):
        """
            returns 2d arrary shaped (batch, node_ids)
            here the number of node ids may vary between batch items
        """
        if isinstance(data.graph, List):
            all_ids = []
            for g in data.graph:
                ids = self.get_node_ids_from_graph(g)
                all_ids.append(ids)
            return all_ids
        else:
            return [self.get_node_ids_from_graph(data.graph)]

    def get_node_ids_from_graph(self, graph):
        # override to make node selection on a subset only
        return list(range(len(graph.ordered_nodes)))  # all nodes
