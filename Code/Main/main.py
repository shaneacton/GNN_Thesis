from torch import nn

from Code.Models.GNNs.Layers.new_layer import NewLayer
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

    layer = NewLayer([768, 5]).to(device)

    gnn.layers = [layer]
    gnn.layer_list = nn.ModuleList(gnn.layers)

    for i, sample in enumerate(samples):
        if i >= 3:
            break

        graph = constructor(sample)
        # render_graph(graph, sample.title_and_peek, reader.datset_name)

        ds_out = gnn(graph)
        print("gnn", gnn, "out:", ds_out, "\n----------------------------------------------------------\n"*2)


# if __name__ == "__main__":
#     from Code.Data.Graph import example_edge_index, example_x
#
#     conv = GATConv(3, 6).to(device)
#
#     out = conv(example_x, example_edge_index)

