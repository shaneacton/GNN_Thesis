if __name__ == "__main__":
    from Datasets.Readers.squad_reader import SQuADDatasetReader
    from Datasets.Readers.qangaroo_reader import QUangarooDatasetReader

    from Code.Config import configs, gcc

    sq_reader = SQuADDatasetReader("SQuAD")
    qangaroo_reader = QUangarooDatasetReader("wikihop")

    # reader = qangaroo_reader
    reader = sq_reader
    samples = reader.get_dev_set()

    constructor = gcc.get_graph_constructor()
    gnn = configs.get_gnn()

    for i, sample in enumerate(samples):
        if i>=3:
            break

        graph = constructor(sample)
        ds_out = gnn(graph)
        print("gnn", gnn, "out:", ds_out, "\n------------------\n")

        graph.render_graph(sample.title_and_peek, reader.datset_name)
