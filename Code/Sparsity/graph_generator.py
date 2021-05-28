import random

import torch

from Code.Training import dev


def get_dense_graph(num_nodes, num_features):
    """
        returns a fully connnected random graph with given num nodes
        to be usable by both a transformer and a gnn, we will create node representations,
        as well as an (E,2) edge list for pytorch geometric
        as well as an (n, n) adjacency matrix to be used as a mask by the transformer
    """
    nodes = torch.rand((num_nodes, num_features)).to(dev())
    froms = []
    tos = []

    for i in range(num_nodes):
        for j in range(num_nodes):
            froms.append(i)
            tos.append(j)

    edge_index = torch.tensor([froms, tos]).to(dev()).long()
    mask = torch.zeros(num_nodes, num_nodes, dtype=torch.bool).to(dev())

    return nodes, edge_index, mask


def get_sparse_graph(num_nodes, num_features, sparsity):
    """
        sparsity is between 0 and 1, and dictates how many of the possible connections are used
    """
    nodes = torch.rand((num_nodes, num_features)).to(dev())
    froms = []
    tos = []

    mask = torch.zeros(num_nodes, num_nodes, dtype=torch.bool).to(dev())

    for i in range(num_nodes):
        for j in range(num_nodes):
            if random.random() < sparsity:  # include this edge
                froms.append(i)
                tos.append(j)
            else:
                mask[i, j] = True  # this connection has been dropped. we should mask it

    edge_index = torch.tensor([froms, tos]).to(dev()).long()

    return nodes, edge_index, mask