import argparse
import copy

import numpy as np
import pandas as pd
from tqdm import tqdm

from Code.HDE.Graph.graph import HDEGraph
from Code.Training import dev
from Config.config import set_conf_files, get_config

embedder = None
CHUNK_SIZE = 5000  # -1 to load whole dataset at once
DATASET_LENGTH = 41000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', '-d', help='Whether or not to run the debug configs - y/n', default="n")
    parser.add_argument('--processed_data_path', '-p', help='Where processed graphs are stored', default="")

    args = parser.parse_args()
    model_conf = "debug_model"
    train_conf = "debug_train"

    from Config.config import conf

    if model_conf is not None or train_conf is not None:
        if model_conf is None:
            model_conf = conf.model_cfg_name
        if train_conf is None:
            train_conf = conf.train_cfg_name

    set_conf_files(model_conf, train_conf)
    from Config.config import conf
    conf.run_args = args


def get_graphs(dataset, use_special_entities, use_detected_entities, sentence_nodes, compliments, codocs, comentions, example_span=None):
    from Code.Utils.dataset_utils import get_wikihop_graphs
    from Config.config import conf
    global embedder

    conf.set("dataset", dataset)
    conf.set("use_special_entities", use_special_entities)
    conf.set("use_detected_entities", use_detected_entities)
    conf.set("use_sentence_nodes", sentence_nodes)
    conf.set("use_codocument_edges", codocs)
    conf.set("use_compliment_edges", compliments)
    conf.set("use_comention_edges", comentions)
    conf.set("bidirectional_edge_types", False)

    try:
        graphs = get_wikihop_graphs(example_span=example_span)
    except:
        print("graphs not generated yet. Generating")
        if embedder is None:
            from Code.Utils.model_utils import get_model_class
            model = get_model_class()().to(dev())
            embedder = copy.deepcopy(model.embedder)
            del model
        graphs = get_wikihop_graphs(embedder=embedder, example_span=example_span)
    return graphs


def get_graph_data(**kwargs):
    cross_dot_ratios, densities, num_nodes, _ = get_graph_stats(**kwargs)

    data = np.array([densities, cross_dot_ratios, num_nodes]).transpose()
    data = pd.DataFrame(data, columns=['Edge Density', 'Cross Document Ratio', "Number of Nodes"])
    return data.replace(0, 0)


def get_graph_stats(**kwargs):
    densities = []
    cross_dot_ratios = []
    num_nodes = []
    all_token_lengths = []
    sum_token_lengths = []
    num_documents = []

    def get_metrics(example_span):
        graphs = get_graphs(example_span=example_span, **kwargs)

        for graph in tqdm(graphs):
            graph: HDEGraph = graph
            densities.append(graph.get_edge_density())
            cross_dot_ratios.append(graph.get_cross_doc_ratio())
            num_nodes.append(len(graph.ordered_nodes))
            all_token_lengths.extend(graph.example.doc_token_lengths)
            sum_token_lengths.append(sum(graph.example.doc_token_lengths))
            num_documents.append(len(graph.doc_nodes))

    if CHUNK_SIZE == -1 or get_config().max_examples != -1:
        get_metrics(None)
    else:
        start = 0
        while start <= DATASET_LENGTH:
            end = start + CHUNK_SIZE
            get_metrics((start, end))
            start += CHUNK_SIZE
    return cross_dot_ratios, densities, num_nodes, (all_token_lengths, sum_token_lengths, num_documents)
