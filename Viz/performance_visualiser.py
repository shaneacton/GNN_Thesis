from math import floor
from typing import List

import nlp
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm


def convert_accs(independents, accs, bin_size=5):
    bins = {}  #  (bin_id -> (total_value, num_observations))
    unique_bins = []
    ind_val2bin = lambda ind: floor(ind/bin_size)
    for i, ind in enumerate(independents):
        bin = ind_val2bin(ind)
        if bin not in bins:
            bins[bin] = [0, 0]
            unique_bins.append(bin)
        bins[bin][0] += accs[i]
        bins[bin][1] += 1

    accs = [bins[bin][0]/bins[bin][1] for bin in unique_bins]
    unique_bins = [b*bin_size for b in unique_bins]
    return unique_bins, accs


def plot(results, independent, dependent):
    # if dependent == "acc":
    independents, dependents = convert_accs(results[independent], results[dependent])
    # else:
    #     independents, dependents = results[independent], results[dependent]
    plt.scatter(independents, dependents, color="blue")

    plt.title(independent + " vs " + dependent + " Correlation. n=" + repr(len(independents)))
    plt.xlabel(independent)
    plt.ylabel(dependent)

    plt.show()


def get_conf_model():
    from Code.Main.sys_arg_launcher import get_args
    args = get_args()

    from Code.Main.run_utils import get_model_config
    conf = get_model_config(args.model_conf, args.train_conf, 0, args)

    from Code.Utils.model_utils import get_model
    model, optimizer, scheduler = get_model(conf.model_name)
    return model


if __name__ == "__main__":
    model = get_conf_model()
    print("loaded model:", model)

    from Code.Utils.dataset_utils import get_wikihop_graphs
    from Code.HDE.Graph.graph import HDEGraph
    graphs: List[HDEGraph] = get_wikihop_graphs(embedder=model.embedder, split=nlp.Split.TEST)

    print("loaded", len(graphs), "graphs")

    from Code.Embedding.bert_embedder import TooManyTokens
    from Code.Embedding.glove_embedder import NoWordsException
    from Code.HDE.hde_model import PadVolumeOverflow, TooManyEdges

    num_discarded = 0
    results = {"loss": [], "acc": [], "nodes": [], "edges": [], "cands": [], "docs":[]}
    for i, graph in tqdm(enumerate(graphs)):
        graph: HDEGraph = graph
        try:
            with torch.no_grad():
                loss, predicted = model(graph=graph)

            results["loss"].append(loss.item())
            results["acc"].append(1 if graph.example.answer==predicted else 0)

            results["nodes"].append(graph.num_nodes)
            results["edges"].append(graph.num_edges)
            results["cands"].append(len(graph.candidate_nodes))
            results["docs"].append(len(graph.doc_nodes))

        except (NoWordsException, PadVolumeOverflow, TooManyEdges, TooManyTokens) as ne:
            num_discarded += 1
            continue

    plot(results, "cands", "loss")