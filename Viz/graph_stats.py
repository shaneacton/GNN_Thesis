import pandas as pd
from matplotlib import pyplot as plt

import seaborn as sns

from Viz.graph_fetch_utils import parse_args, get_graph_data, get_graph_stats


ROWS = None
COLS = None

BINS_FAC = 40

node_variants = ["Detected Entities", "Special and Detected Entities", "Sentence Nodes"]
edge_variants = ["CoDocument Edges", "No Comention Edges"]


def plot_stat_hist(configs, config_name, row, col, stat_name, x_range=None):
    stats = get_graph_data(**configs[config_name])
    if ROWS == 1:
        axis = AXES[col]
    elif COLS == 1:
        axis = AXES[row]
    else:
        axis = AXES[row, col]
    bins = 15

    if x_range is not None:
        axis.set(xlim=x_range)
        value_range = stats[stat_name].max() - stats[stat_name].min()
        bins = max(1, int(BINS_FAC * value_range / x_range[1]))

    plot = sns.histplot(ax=axis, data=stats, multiple="stack", x=stat_name, bins=bins)
    plot.set_xlabel(plot.xaxis.get_label().get_text(), fontsize=15)
    plot.set_ylabel(plot.yaxis.get_label().get_text(), fontsize=15)
    axis.set_title(config_name)


def plot_stats_boxes(configs, stat_name, ordered_confs):
    relevant_stats = pd.DataFrame()
    new_subfigs(1, 1, width=10)
    ordered_confs = ["Default"] + ordered_confs

    for i, conf in enumerate(ordered_confs):
        stats = get_graph_data(**configs[conf])
        stat = stats[stat_name]
        print("stat:", stat)
        relevant_stats[conf] = stat
    sns.boxplot(ax=AXES, data=relevant_stats, orient="h", showfliers=False)  # RUN PLOT
    plt.title(stat_name)
    plt.tight_layout()
    plt.show()


def new_subfigs(rows, cols, width=15, font_scale=1.4):
    global FIG, AXES, ROWS, COLS
    ROWS = rows
    COLS = cols
    sns.set(font_scale=font_scale)
    FIG, AXES = plt.subplots(rows, cols, figsize=(width, 2 + rows*3.5))
    plt.subplots_adjust(hspace=0.75, wspace=0.3)


def plot_token_counts(**kwargs):
    new_subfigs(4, 1, width=14)
    _, _, _, (all_token_lengths, sum_token_lengths, num_documents, num_candidates) = get_graph_stats(**kwargs)
    ax1, ax2, ax3, ax4 = AXES[0], AXES[1], AXES[2], AXES[3]
    doc_data = pd.DataFrame({"Document Token Lengths": all_token_lengths})
    total_data = pd.DataFrame({"Wikihop Datapoint Total Tokens": sum_token_lengths})
    num_docs = pd.DataFrame({"Wikihop Datapoint Document Count": num_documents})
    num_cands = pd.DataFrame({"Wikihop Datapoint Candidate Count": num_candidates})

    sns.boxplot(ax=ax1, data=num_docs, orient="h", showfliers=False)
    sns.boxplot(ax=ax2, data=doc_data, orient="h", showfliers=False)
    sns.boxplot(ax=ax3, data=total_data, showfliers=False, orient="h")
    sns.boxplot(ax=ax4, data=num_cands, showfliers=False, orient="h")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parse_args()

    configs = {"Default":           {"dataset": "wikihop", "use_special_entities": True, "use_detected_entities": False,
                    "sentence_nodes": False, "compliments": False, "codocs": False, "comentions": True},
               "CoDocument Edges":  {"dataset": "wikihop", "use_special_entities": True, "use_detected_entities": False,
                    "sentence_nodes": False, "compliments": False, "codocs": True, "comentions": True},
               "Compliment Edges":  {"dataset": "wikihop", "use_special_entities": True, "use_detected_entities": False,
                    "sentence_nodes": False, "compliments": True, "codocs": False, "comentions": True},
               "No Comention Edges": {"dataset": "wikihop", "use_special_entities": True, "use_detected_entities": False,
                    "sentence_nodes": False, "compliments": False, "codocs": False, "comentions": False},
               "Detected Entities": {"dataset": "wikihop", "use_special_entities": False, "use_detected_entities": True,
                    "sentence_nodes": False, "compliments": False, "codocs": False, "comentions": True},
               "Special and Detected Entities": {"dataset": "wikihop", "use_special_entities": True, "use_detected_entities": True,
                    "sentence_nodes": False, "compliments": False, "codocs": False, "comentions": True},
               "Sentence Nodes":    {"dataset": "wikihop", "use_special_entities": True, "use_detected_entities": False,
                    "sentence_nodes": True, "compliments": False, "codocs": False, "comentions": True}
               }

    plot_token_counts(**configs["Default"])
    # plot_stats_boxes(configs, "Number of Nodes", node_variants)
    # plot_stats_boxes(configs, 'Cross Document Ratio', node_variants + edge_variants)
    # plot_stats_boxes(configs, 'Edge Density', node_variants + edge_variants)

