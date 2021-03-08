from math import ceil
from typing import List

import torch
from torch_geometric.nn.pool.topk_pool import topk, filter_adj

from Code.Training import device
from Config.config import conf


class SAGPool(torch.nn.Module):
    r"""The self-attention pooling operator from the `"Self-Attention Graph
    Pooling" <https://arxiv.org/abs/1904.08082>`_ and `"Understanding
    Attention and Generalization in Graph Neural Networks"
    <https://arxiv.org/abs/1905.02850>`_ papers

    if :obj:`min_score` :math:`\tilde{\alpha}` is :obj:`None`:

        .. math::
            \mathbf{y} &= \textrm{GNN}(\mathbf{X}, \mathbf{A})

            \mathbf{i} &= \mathrm{top}_k(\mathbf{y})

            \mathbf{X}^{\prime} &= (\mathbf{X} \odot
            \mathrm{tanh}(\mathbf{y}))_{\mathbf{i}}

            \mathbf{A}^{\prime} &= \mathbf{A}_{\mathbf{i},\mathbf{i}}


    where nodes are dropped based on a learnable projection score
    :math:`\mathbf{p}`.
    Projections scores are learned based on a graph neural network layer.

    Args:
        in_channels (int): Size of each input sample.
        ratio (float): Graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`.
            This value is ignored if min_score is not None.
            (default: :obj:`0.5`)
        GNN (torch.nn.Module, optional): A graph neural network layer for
            calculating projection scores (one of
            :class:`torch_geometric.nn.conv.GraphConv`,
            :class:`torch_geometric.nn.conv.GCNConv`,
            :class:`torch_geometric.nn.conv.GATConv` or
            :class:`torch_geometric.nn.conv.SAGEConv`). (default:
            :class:`torch_geometric.nn.conv.GraphConv`)
        multiplier (float, optional): Coefficient by which features gets
            multiplied after pooling. This can be useful for large graphs and
            when :obj:`min_score` is used. (default: :obj:`1`)
        nonlinearity (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.tanh`)
        **kwargs (optional): Additional parameters for initializing the graph
            neural network layer.
    """
    def __init__(self, in_channels, GNN_CLASS=None, min_score=None,
                 multiplier=1, nonlinearity=torch.tanh, **kwargs):
        if GNN_CLASS is None:
            from Code.Training.Utils.model_utils import GNN_MAP
            GNN_CLASS = GNN_MAP[conf.pool_class]

        super(SAGPool, self).__init__()

        self.in_channels = in_channels
        self.ratio = conf.pool_ratio
        self.gnn = GNN_CLASS(in_channels, 1, **kwargs)
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        self.reset_parameters()

    def reset_parameters(self):
        self.gnn.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, excluded_nodes=None):
        """
        A variation of the SAGPool to allow nodes to be excluded from pooling.
        pools the selected ratio from only eligible nodes,
        thus the effective number of pooled nodes will differ from the defined ratio
        """
        batch = edge_index.new_zeros(x.size(0))
        if excluded_nodes is None:
            excluded_nodes = torch.tensor([]).to(device).long()
        elif isinstance(excluded_nodes, List):
            excluded_nodes = torch.tensor(excluded_nodes).to(device).long()

        attn = x
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn
        score = self.gnn(attn, edge_index).view(-1)

        score = self.nonlinearity(score)
        effective_score = score.clone()
        effective_score[excluded_nodes] = -1.  # ensures the excluded won't be selected for saving
        # print("x:", x.size(), "num excl:", excluded_nodes.size())

        effective_ratio = self.get_effective_pooling_ratio(x, excluded_nodes)
        perm = topk(effective_score, effective_ratio, batch, self.min_score)

        """
            if the pooling ratio is high enough, the pooler will select excluded nodes to 
            be saved, despite having negative scores
            so we should take care not to double add these nodes
        """

        excluded_nodes = {n.item() for n in excluded_nodes} - {n.item() for n in perm}  # remove already accounted
        excluded_nodes = sorted(list(excluded_nodes))
        excluded_nodes = torch.tensor(excluded_nodes).long().to(device)

        perm = torch.cat([perm, excluded_nodes])  # excluded nodes are added back to the saved list
        x = x[perm] * score[perm].view(-1, 1)
        # print("ratio:", self.ratio, "eff ratio:", effective_ratio, "x:", x.size())

        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm,
                                           num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm, score[perm]

    def get_effective_pooling_ratio(self, x, excluded_nodes):
        """
                    all excluded will be pooled by default, and added back in later.
                    thus reducing the number of actually pooled nodes.
                    To correct this, we need to pool more nodes than expected
                """
        num_excluded = excluded_nodes.size(0)
        num_pool_candidates = x.size(0) - num_excluded  # the nodes which can be pooled/ not excluded
        num_to_keep = ceil(self.ratio * num_pool_candidates) + num_excluded
        effective_ratio = num_to_keep / x.size(0)
        # print("effective ratio:", effective_ratio)
        return effective_ratio

    def __repr__(self):
        return '{}({}, {}, {}={}, multiplier={})'.format(
            self.__class__.__name__, self.gnn.__class__.__name__,
            self.in_channels,
            'ratio' if self.min_score is None else 'min_score',
            self.ratio if self.min_score is None else self.min_score,
            self.multiplier)
