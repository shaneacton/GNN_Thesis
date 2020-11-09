import time

from torch_geometric.nn import GATConv

from Code.Models.GNNs.ContextGNNs.geometric_context_gnn import GeometricContextGNN


class ContextGAT(GeometricContextGNN):

    def init_layers(self, in_features, num_layers=5) -> int:
        return super().init_layers(in_features, GATConv, num_layers=num_layers)


if __name__ == "__main__":
    from Code.Config import gec, gnnc
    from Code.Config import gcc
    from Code.Test.examples import test_example

    embedder = gec.get_graph_embedder(gcc)

    gat = ContextGAT(embedder, gnnc)

    print(test_example)

    out = gat(test_example)

    print("done")
    stime = time.time()
    out = gat(test_example)

    print("f time:", (time.time()-stime))