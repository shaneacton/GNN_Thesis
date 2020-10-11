import copy

from Viz.context_graph_visualiser import render_graph

if __name__ == "__main__":
    from Datasets.Readers.squad_reader import SQuADDatasetReader
    from Datasets.Readers.qangaroo_reader import QUangarooDatasetReader

    from Code.Config import configs, gcc, vizconf

    sq_reader = SQuADDatasetReader("SQuAD")
    qangaroo_reader = QUangarooDatasetReader("wikihop")

    # reader = qangaroo_reader
    reader = sq_reader
    samples = reader.get_dev_set()

    gcc.context_max_chars = vizconf.max_context_graph_chars
    constructor = gcc.get_graph_constructor()

    for i, sample in enumerate(samples):
        if i < 0:
            continue
        if i >= 6:
            break

        graph = constructor(sample)
        render_graph(graph, sample.title_and_peek + repr(i), reader.datset_name)
        print("rendered graph", sample.title_and_peek + repr(i))

