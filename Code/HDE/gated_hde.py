import time

import torch

from Code.GNNs.gnn_stack import GNNStack
from Code.GNNs.hde_gnn import HDEGNN
from Code.HDE.hde_glove import HDEGlove
from Code.HDE.hde_model import TooManyEdges
from Code.HDE.wikipoint import Wikipoint
from Code.Training import device
from Config.config import conf


class GatedHDE(HDEGlove):

    def __init__(self, **kwargs):
        super().__init__(GNNClass=None, **kwargs)
        self.gnn = GNNStack(HDEGNN)

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

        num_edges = len(graph.unique_edges)
        if num_edges > conf.max_edges != -1:
            raise TooManyEdges()

        t = time.time()
        x = self.gnn(x, graph)

        if conf.print_times:
            print("passed gnn in", (time.time() - t))
        t = time.time()

        # x has now been transformed by the GNN layers. Must map to  a prob dist over candidates
        final_probs = self.pass_output_model(x, example, graph)
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