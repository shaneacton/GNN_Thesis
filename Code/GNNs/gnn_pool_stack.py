from typing import List

from torch import nn
from torch.nn import ModuleList

from Code.GNNs.gated_gnn import GatedGNN
from Code.GNNs.gnn_stack import GNNLayer
from Config.config import conf


class GNNPoolStack(nn.Module):

    def __init__(self, GNNClass, PoolClass, use_gating=False, **layer_kwargs):
        super().__init__()
        layers = []
        poolers = []
        for layer_i in range(conf.num_layers):
            in_size = conf.embedded_dims if layer_i == 0 else conf.hidden_size
            layer = GNNLayer(GNNClass, in_size, **layer_kwargs)
            if use_gating:
                layer = GatedGNN(layer)
            layers.append(layer)

            pooler = PoolClass(in_size)
            poolers.append(pooler)

        self.poolers = ModuleList(poolers)
        self.layers = ModuleList(layers)

    def forward(self, x, edge_index, cand_idxs, node_id_map=None):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            x, edge_index, node_id_map = self.pool(self.poolers[i], x, edge_index, node_id_map, cand_idxs)
        return x, edge_index, node_id_map

    @staticmethod
    def pool(pooler, x, edge, previous_node_id_map=None, excluded_nodes: List = None):
        """
        if this is not the first pooling, the previous map must be provided, to relate the new ids to the original
        previous maps original_ids -> last_pooled_ids
        new map maps last_pooled_ids -> newest_pooled_ids

        excluded nodes ids must reference original graph. they will be converted to new graph ids
        """
        if excluded_nodes is None:
            excluded_nodes = []
        if excluded_nodes is None:
            raise Exception()

        effective_excluded_nodes = excluded_nodes
        if previous_node_id_map is not None:
            effective_excluded_nodes = [previous_node_id_map[e] for e in excluded_nodes if e in previous_node_id_map.keys()]

        x, edge, _, _, perm, _ = pooler(x, edge, excluded_nodes=effective_excluded_nodes)
        new_node_id_map = {}  # maps previous node ids, to new node ids
        for after_i, before_i in enumerate(perm):
            # print("before:", before_i, "after:", after_i)
            new_node_id_map[before_i.item()] = after_i

        if previous_node_id_map is not None:  # creates a map from original node ids to  new node ids
            full_map = {}
            for orig_id in previous_node_id_map.keys():
                prev_id = previous_node_id_map[orig_id]
                if prev_id in new_node_id_map:  # else this node has been pooled
                    new_id = new_node_id_map[prev_id]
                    full_map[orig_id] = new_id
            new_node_id_map = full_map

        return x, edge, new_node_id_map