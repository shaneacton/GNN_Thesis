import argparse
import copy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from Code.HDE.Graph.graph import HDEGraph
from Code.Training import dev
from Config.config import set_conf_files, get_config
import seaborn as sns

DATASET = "wikihop"
SPECIAL_ENTITIES = True
DETECTED_ENTITIES = False
SENTENCE_NODES = False

BIDIRECTIONAL_EDGES = False
CODOCUMENT_EDGES = False
COMPLIMENT_EDGES = False

embedder = None


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
    conf.set("bidirectional_edge_types", BIDIRECTIONAL_EDGES)
    conf.set("use_codocument_edges", codocs)
    conf.set("use_compliment_edges", compliments)
    conf.set("use_comention_edges", comentions)

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


def get_graph_stats(dataset, use_special_entities, use_detected_entities, sentence_nodes, compliments, codocs, comentions):
    densities = []
    cross_dot_ratios = []

    def get_metrics(example_span):
        graphs = get_graphs(dataset, use_special_entities, use_detected_entities, sentence_nodes,
                            compliments, codocs, comentions, example_span=example_span)

        for graph in tqdm(graphs):
            graph: HDEGraph = graph
            densities.append(graph.get_edge_density())
            cross_dot_ratios.append(graph.get_cross_doc_ratio())

    if CHUNK_SIZE == -1 or get_config().max_examples != -1:
        get_metrics(None)
    else:
        start = 0
        while start <= DATASET_LENGTH:
            end = start + CHUNK_SIZE
            get_metrics((start, end))
            start += CHUNK_SIZE

    data = np.array([densities, cross_dot_ratios]).transpose()
    data = pd.DataFrame(data, columns=['Edge Density', 'Cross Document Ratio'])
    data["Dataset"] = dataset
    data["Special Entities"] = use_special_entities
    data["Detected Entities"] = use_detected_entities
    return data.replace(0, 0)


# def plot_wikimed_stats(use_special_entities, use_detected_entities, row=0):
#     wiki_stats = get_graph_stats("wikihop", use_special_entities, use_detected_entities)
#     med_stats = get_graph_stats("medhop", use_special_entities, use_detected_entities)
#
#     stats = pd.concat([wiki_stats, med_stats])
#
#     sns.kdeplot(ax=AXES[row, 0], data=stats, multiple="stack", x="Edge Density", hue='Dataset')
#     sns.histplot(ax=AXES[row, 1], data=stats, multiple="stack", x="Cross Document Ratio", hue='Dataset', stat="density", log_scale=True, bins=15)
#     AXES[row, 0].set_title("Edge Density")
#     AXES[row, 1].set_title("Cross Document Ratio")


def plot_stats(dataset, use_special_entities, use_detected_entities, sentence_nodes, compliments, codocs, comentions, row=0, title=None):
    stats = get_graph_stats(dataset, use_special_entities, use_detected_entities, sentence_nodes, compliments, codocs, comentions)
    print("stats:", stats)
    if ROWS == 1:
        ax0 = AXES[0]
        ax1 = AXES[1]
    else:
        ax0 = AXES[row, 0]
        ax1 = AXES[row, 1]
    ax0.set(xlim=EDGE_DENSITY_SPAN)
    cross_doc_range = stats["Cross Document Ratio"].max() - stats["Cross Document Ratio"].min()
    # bins = int(100*cross_doc_range/max_range)
    bins = max(1, int(BINS_FAC*cross_doc_range/MAX_CDR_RANGE))
    print(title, "range:", cross_doc_range, "bins:", bins)
    ax1.set(xlim=(-5, MAX_CDR_RANGE))
    ed = sns.histplot(ax=ax0, data=stats, multiple="stack", x="Edge Density", bins=15)
    cdr = sns.histplot(ax=ax1, data=stats, multiple="stack", x="Cross Document Ratio", bins=bins)

    ed.set_xlabel(ed.xaxis.get_label().get_text(), fontsize=15)
    ed.set_ylabel(ed.yaxis.get_label().get_text(), fontsize=15)
    cdr.set_xlabel(cdr.xaxis.get_label().get_text(), fontsize=15)
    cdr.set_ylabel(cdr.yaxis.get_label().get_text(), fontsize=15)

    if title is None:
        ax0.set_title("Edge Density")
        ax1.set_title("Cross Document Ratio")
    else:
        ax0.set_title(title)
        ax1.set_title(title)


ROWS = 3
MAX_CDR_RANGE = 400
# MAX_CDR_RANGE = 900
EDGE_DENSITY_SPAN = (0.02, 0.4)

BINS_FAC = 40

CHUNK_SIZE = 5000  # -1 to load whole dataset at once
DATASET_LENGTH = 41000


FIG, AXES = plt.subplots(ROWS, 2, figsize=(15, 3 + ROWS*3))
plt.subplots_adjust(hspace=0.5, wspace=0.3)
sns.set(font_scale=1.75)

if __name__ == "__main__":
    parse_args()
    # AXES[0, 1].set_xscale('symlog')
    FIG.suptitle('Title')

    # plot_wikimed_stats(SPECIAL_ENTITIES, DETECTED_ENTITIES)
    # plot_wikimed_stats(not SPECIAL_ENTITIES, not DETECTED_ENTITIES, row=1)

    # plot_stats("wikihop", SPECIAL_ENTITIES, DETECTED_ENTITIES, SENTENCE_NODES, False, False, True, title="Default")
    # plot_stats("wikihop", SPECIAL_ENTITIES, DETECTED_ENTITIES, SENTENCE_NODES, False, True, True, title="CoDocument Edges", row=1)
    # plot_stats("wikihop", SPECIAL_ENTITIES, DETECTED_ENTITIES, SENTENCE_NODES, True, False, True, title="Compliment Edges", row=2)
    # plot_stats("wikihop", SPECIAL_ENTITIES, DETECTED_ENTITIES, SENTENCE_NODES, False, False, False, title="No Comention Edges", row=3)

    plot_stats("wikihop", True, False, False, False, False, True, title="Special Entities")
    plot_stats("wikihop", False, True, False, False, False, True, title="Detected Entities", row=1)
    plot_stats("wikihop", True, False, True, False, False, True, title="Sentence Nodes", row=2)


    plt.show()
