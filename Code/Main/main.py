import math

from torch import nn

from Code.Models.GNNs.Layers.graph_layer import GraphLayer
from Code.Training import device
from Viz.graph_visualiser import render_graph

if __name__ == "__main__":
    from Datasets.Readers.squad_reader import SQuADDatasetReader
    from Datasets.Readers.qangaroo_reader import QUangarooDatasetReader

    from Code.Config import configs, gcc

    sq_reader = SQuADDatasetReader("SQuAD")
    qangaroo_reader = QUangarooDatasetReader("wikihop")

    reader = qangaroo_reader
    # reader = sq_reader
    samples = reader.get_dev_set()

    constructor = gcc.get_graph_constructor()
    gnn = configs.get_gnn()

    for i, sample in enumerate(samples):
        if i < 0:
            continue
        if i >= 6:
            break

        graph = constructor(sample)
        # render_graph(graph, sample.title_and_peek + repr(i), reader.datset_name)

        ds_out = gnn(graph)
        print("out:", ds_out)
        num_edges = ds_out.edge_index.size(1)
        num_nodes = ds_out.node_types.size(0)
        print("num_nodes:", num_nodes, "num edges:", num_edges, " scale fac:", math.log(num_edges, num_nodes))
        print("y' out:", ds_out.x.view(-1), "\n----------------------------------------------------------\n"*2)


    print("gnn", gnn)
    # print("num params in layer:", layer.num_params)


# if __name__ == "__main__":
#     from Code.Data.Graph import example_edge_index, example_x
#
#     conv = GATConv(3, 6).to(device)
#
#     out = conv(example_x, example_edge_index)

