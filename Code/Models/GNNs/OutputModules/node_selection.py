from typing import List

import torch
from torch import nn

from Code.Models.GNNs.OutputModules.output_model import OutputModel
from Code.Training import device


class NodeSelection(OutputModel):
    """predicts a probability for all/subset of the nodes"""

    def __init__(self, in_features):
        super().__init__(in_features)
        self.probability_mapper = nn.Linear(in_features, 1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, data, node_ids=None, inplace=True):
        """
        :param node_ids: optional override to which nodes are candidates
        :param inplace: if  true, alters the datapoints x val. if false just returns the result
        :return: data if inplace=true, else predictions
        """
        if node_ids is None:
            batch_node_ids = self.get_batched_node_ids(data)
        if not isinstance(node_ids, torch.Tensor):
            probs = []
            max_node_count = -1
            # must do final probability mapping separately due to differing classification node counts per batch item
            for graph_node_ids in batch_node_ids:
                node_ids = torch.tensor(graph_node_ids).to(device)
                # print("selecting node:", node_ids, "\nfrom", data.x.size())
                choices = torch.index_select(data.x, 0, node_ids)
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

            # print("x:",data.x.size(), "choices:", choices.size())
            # print("x before:", data.x)
            # print("probabilities", batchwise_probabilities.size(), batchwise_probabilities)
            if inplace:
                data.x = batchwise_probabilities
                # print("x after:", data.x)
                return data
            else:
                return batchwise_probabilities

    def get_batched_node_ids(self, data):
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
        return list(range(len(graph.ordered_nodes)))