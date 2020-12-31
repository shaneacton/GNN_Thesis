import os
import sys
import time

# from torch_geometric.nn import GATConv

from Code.Models.GNNs.Custom.asymmetrical_gat import AsymGat

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
dir_path_1 = os.path.join(dir_path_1, "..", "..")
sys.path.append(dir_path_1)
sys.path.append(os.path.join(dir_path_1, 'Code'))

from Code.Models.GNNs.ContextGNNs.geometric_context_gnn import GeometricContextGNN


class ContextGAT(GeometricContextGNN):

    def init_layers(self, in_features) -> int:
        return super().init_layers(in_features, AsymGat)


if __name__ == "__main__":

    from Code.Config import gec, gnnc
    from Code.Config import gcc
    from Code.Play.examples import test_example

    embedder = gec.get_graph_embedder(gcc)

    gat = ContextGAT(embedder, gnnc)

    print(test_example)

    out = gat(test_example)

    print("done")
    stime = time.time()
    out = gat(test_example)

    print("f time:", (time.time()-stime))