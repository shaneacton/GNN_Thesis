import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from Code.HDE.Graph.graph import HDEGraph
from Code.Training import dev
from Config.config import set_conf_files
import seaborn as sns

DATASET = "wikihop"
SPECIAL_ENTITIES = True
DETECTED_ENTITIES = False
SENTENCE_NODES = False

BIDIRECTIONAL_EDGES = False
CODOCUMENT_EDGES = False
COMPLIMENT_EDGES = False


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


def get_graphs(dataset, use_special_entities, use_detected_entities, sentence_nodes, compliments, codocs):
    from Code.Utils.dataset_utils import get_wikihop_graphs
    from Config.config import conf

    conf.set("dataset", dataset)
    conf.set("use_special_entities", use_special_entities)
    conf.set("use_detected_entities", use_detected_entities)
    conf.set("use_sentence_nodes", sentence_nodes)
    conf.set("bidirectional_edge_types", BIDIRECTIONAL_EDGES)
    conf.set("use_codocument_edges", codocs)
    conf.set("use_compliment_edges", compliments)

    try:
        graphs = get_wikihop_graphs()
    except:
        print("graphs not generated yet. Generating")
        from Code.Utils.model_utils import get_model_class
        model = get_model_class()().to(dev())
        graphs = get_wikihop_graphs(embedder=model.embedder)

    return graphs


def get_graph_stats(dataset, use_special_entities, use_detected_entities, sentence_nodes, compliments, codocs):
    graphs = get_graphs(dataset, use_special_entities, use_detected_entities, sentence_nodes, compliments, codocs)

    densities = []
    cross_dot_ratios = []
    for graph in tqdm(graphs):
        graph: HDEGraph = graph
        densities.append(graph.get_edge_density())
        cross_dot_ratios.append(graph.get_cross_doc_ratio())

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


def plot_stats(dataset, use_special_entities, use_detected_entities, sentence_nodes, compliments, codocs, row=0, title=None):
    stats = get_graph_stats(dataset, use_special_entities, use_detected_entities, sentence_nodes, compliments, codocs)
    print("stats:", stats)
    if ROWS == 1:
        ax0 = AXES[0]
        ax1 = AXES[1]
    else:
        ax0 = AXES[row, 0]
        ax1 = AXES[row, 1]
    ax0.set(xlim=(0, 0.35))
    cross_doc_range = stats["Cross Document Ratio"].max() - stats["Cross Document Ratio"].min()
    # bins = int(100*cross_doc_range/max_range)
    bins = int(BINS_FAC*cross_doc_range/MAX_CDR_RANGE)
    print(title, "range:", cross_doc_range, "bins:", bins)
    ax1.set(xlim=(0, MAX_CDR_RANGE))
    sns.histplot(ax=ax0, data=stats, multiple="stack", x="Edge Density", hue='Dataset', bins=15)
    sns.histplot(ax=ax1, data=stats, multiple="stack", x="Cross Document Ratio", hue='Dataset', bins=bins)
    if title is None:
        ax0.set_title("Edge Density")
        ax1.set_title("Cross Document Ratio")
    else:
        ax0.set_title(title)
        ax1.set_title(title)


ROWS = 3
# MAX_CDR_RANGE = 400
MAX_CDR_RANGE = 900

BINS_FAC = 40


FIG, AXES = plt.subplots(ROWS, 2, figsize=(15, 3 + ROWS*2.5))
plt.subplots_adjust(hspace=0.5, wspace=0.4)

if __name__ == "__main__":
    parse_args()
    # AXES[0, 1].set_xscale('symlog')
    FIG.suptitle('Title')

    # plot_wikimed_stats(SPECIAL_ENTITIES, DETECTED_ENTITIES)
    # plot_wikimed_stats(not SPECIAL_ENTITIES, not DETECTED_ENTITIES, row=1)

    plot_stats("wikihop", SPECIAL_ENTITIES, DETECTED_ENTITIES, SENTENCE_NODES, False, False, title="Default")
    plot_stats("wikihop", SPECIAL_ENTITIES, DETECTED_ENTITIES, SENTENCE_NODES, False, True, title="CoDocument Edges", row=1)
    plot_stats("wikihop", SPECIAL_ENTITIES, DETECTED_ENTITIES, SENTENCE_NODES, True, False, title="Compliment Edges", row=2)

    # plot_stats("wikihop", True, False, False, False, False, title="Special Entities")
    # plot_stats("wikihop", False, True, False, False, False, title="Detected Entities", row=1)
    # plot_stats("wikihop", True, False, True, False, False, title="Sentence Nodes", row=2)


    plt.show()
