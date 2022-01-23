import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from Code.HDE.Graph.graph import HDEGraph
from Config.config import set_conf_files
import seaborn as sns

DATASET = "wikihop"
SPECIAL_ENTITIES = True
DETECTED_ENTITIES = False


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


def get_graphs(dataset, use_special_entities, use_detected_entities):
    from Code.Utils.dataset_utils import get_wikihop_graphs
    from Config.config import conf

    conf.set("dataset", dataset)
    conf.set("use_special_entities", use_special_entities)
    conf.set("use_detected_entities", use_detected_entities)

    graphs = get_wikihop_graphs()
    return graphs


def get_graph_stats(dataset, use_special_entities, use_detected_entities):
    graphs = get_graphs(dataset, use_special_entities, use_detected_entities)

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
    return data


def plot_stats(use_special_entities, use_detected_entities, row=0):
    wiki_stats = get_graph_stats("wikihop", use_special_entities, use_detected_entities)
    med_stats = get_graph_stats("medhop", use_special_entities, use_detected_entities)

    stats = pd.concat([wiki_stats, med_stats]).replace(0, 0.01)

    sns.kdeplot(ax=AXES[row, 0], data=stats, multiple="stack", x="Edge Density", hue='Dataset')
    sns.histplot(ax=AXES[row, 1], data=stats, multiple="stack", x="Cross Document Ratio", hue='Dataset', stat="density", log_scale=True)
    AXES[row, 0].set_title("Edge Density")
    AXES[row, 1].set_title("Cross Document Ratio")


FIG, AXES = plt.subplots(2, 2, figsize=(15, 5))
if __name__ == "__main__":
    parse_args()
    AXES[0, 1].set_xscale('symlog')
    FIG.suptitle('Title')

    plot_stats(SPECIAL_ENTITIES, DETECTED_ENTITIES)
    plot_stats(not SPECIAL_ENTITIES, not DETECTED_ENTITIES)




    plt.show()
