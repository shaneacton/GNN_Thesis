import torch
from torch import nn

from Code.Models.GNNs.OutputModules.output_model import OutputModel
from Code.Training import device


class NodeSelection(OutputModel):
    """predicts a probability for all/subset of the nodes"""

    def __init__(self, in_features):
        super().__init__()
        self.probability_mapper = nn.Linear(in_features, 1)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, data, node_ids=None):
        if node_ids is None:
            node_ids = self.get_node_ids(data)
        if not isinstance(node_ids, torch.Tensor):
            # print("node ids:",node_ids)
            node_ids = torch.tensor(node_ids).to(device)

        choices = torch.index_select(data.x, 0,node_ids)
        # print("x:",data.x.size(), "choices:", choices.size())
        probabilities = self.probability_mapper(choices)
        # print("x before:", data.x)
        data.x = self.softmax(probabilities).view(-1)
        # print("x after:", data.x)
        return data

    def get_node_ids(self, data):
        # override to make node selection on a subset only
        return list(range(len(data.graph.ordered_nodes)))