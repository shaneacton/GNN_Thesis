import time
from typing import List

import torch
from torch_geometric.nn import GATConv

from Code.GNNs.gnn_pool_stack import GNNPoolStack
from Code.HDE.hde_glove import HDEGlove
from Code.HDE.hde_model import TooManyEdges
from Code.HDE.wikipoint import Wikipoint
from Code.Pooling.custom_sag_pool import SAGPool
from Code.Training import device
from Config.config import conf


class HDEPool(HDEGlove):

    def __init__(self, GNN_CLASS=GATConv, PoolerClass=SAGPool, **kwargs):
        super().__init__(GNNClass=None, **kwargs)
        self.gnn = GNNPoolStack(GNN_CLASS, PoolerClass)

    def forward(self, example: Wikipoint=None, graph=None):
        """
            nodes are created for each support, as well as each candidate and each context entity
            nodes are concattenated as follows: supports, entities, candidates

            nodes are connected according to the HDE paper
            the graph is converted to a pytorch geometric datapoint
        """
        if graph is None:
            graph = self.create_graph(example)
        else:
            example = graph.example
        x = self.get_graph_features(example)

        edge_index = graph.edge_index()
        num_edges = len(graph.unique_edges)
        if num_edges > conf.max_edges != -1:
            raise TooManyEdges()

        t = time.time()
        x, edge_index, node_id_map = self.gnn(x, edge_index, graph.candidate_nodes)

        if conf.print_times:
            print("passed gnn in", (time.time() - t))
        t = time.time()

        # x has now been transformed by the GNN layers. Must map to  a prob dist over candidates
        final_probs = self.pass_output_model(x, example, graph, node_id_map=node_id_map)
        pred_id = torch.argmax(final_probs)
        pred_ans = example.candidates[pred_id]

        if conf.print_times:
            print("passed output model in", (time.time() - t))

        if example.answer is not None:
            ans_id = example.candidates.index(example.answer)
            probs = final_probs.view(1, -1)  # batch dim
            ans = torch.tensor([ans_id]).to(device)
            loss = self.loss_fn(probs, ans)

            return loss, pred_ans

        return pred_ans


