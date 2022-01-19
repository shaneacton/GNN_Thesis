import argparse

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from Code.HDE.Graph.graph import HDEGraph
from Config.config import set_conf_files
import seaborn as sns

DATASET = "wikihop"


if __name__ == "__main__":

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

    conf.set("dataset", DATASET)
    conf.set("max_examples", 100)

    conf.run_args = args

    from Code.Utils.dataset_utils import get_wikihop_graphs
    graphs = get_wikihop_graphs()

    densities = []
    cross_dot_ratios = []
    for graph in tqdm(graphs):
        graph: HDEGraph = graph
        densities.append(graph.get_edge_density())
        cross_dot_ratios.append(graph.get_cross_doc_ratio())

    sns.displot(data=np.array(densities), kde=True)  # , bw=0.25
    plt.xlabel("edge density")
    plt.show()

    sns.displot(data=np.array(cross_dot_ratios), kde=True)  # bw=0.15
    plt.xlabel("cross document ratio")
    plt.show()


