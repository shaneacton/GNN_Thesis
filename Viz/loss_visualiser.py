import copy
import json
from typing import List
import matplotlib.pyplot as plt
import numpy

import os
import sys
from os.path import join

from Checkpoint.checkpoint_utils import loss_plot_path

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
dir_path_1 = join(dir_path_1, "GNN_Thesis")
sys.path.append(dir_path_1)
sys.path.append(os.path.join(dir_path_1, 'Code'))
sys.path.append(os.path.join(dir_path_1, 'Checkpoint'))
sys.path.append(os.path.join(dir_path_1, 'conf'))

from Config.config import conf
from Config.config_utils import load_config


def compare(names, show=True):
    loss_ax, acc_ax = None, None
    colours = ["g", "b", "r", "y"]
    all_names = []
    for i, name_or_group in enumerate(names):
        if not isinstance(name_or_group, List):
            _names = [name_or_group]
        else:
            _names = name_or_group
        for name in _names:
            data = get_training_results(name)
            losses = smooth(data["losses"])
            # print("got losses:", losses)
            # print("from:", save_path)
            train_accs = smooth(data["train_accs"])
            valid_accs = data["valid_accs"]
            loss_ax, acc_ax = plot_loss_and_acc(losses, train_accs, epochs, name, valid_accs=valid_accs,
                                                loss_ax=loss_ax, acc_ax=acc_ax, colour=colours[i])

    plt.legend()

    if show:
        plt.show()
    names = [save_path_to_name(s).replace(" ", "_").replace("hde_", "") for s in save_paths]

    compare_save_path = "/".join(save_paths[0].split("/")[:-1]) + "/compare_"
    compare_save_path += "_".join(names)
    compare_save_path += ".png"
    compare_save_path = compare_save_path
    print("saving compare to:", compare_save_path)
    plt.savefig(compare_save_path)


def save_path_to_name(save_path):
    name = save_path.split("/")[-1].split(".")[0].replace("_losses", "").replace("_", " ")
    return name


def smooth(seq):
    seq = remove_outliers(seq)
    seq = get_rolling_averages(seq)
    return seq


def plot_loss_and_acc(losses, accs, epochs, name, valid_accs=None, fig=None, loss_ax=None, acc_ax=None, colour="g"):
    if fig is None and loss_ax is None:
        fig = plt.figure()

    if loss_ax is None:
        loss_ax = fig.add_subplot(111)
        loss_ax.set_xlabel("Epoch")
        loss_ax.set_ylabel('loss')
        acc_ax = loss_ax.twinx()
        acc_ax.set_ylabel('accuracy')
    loss_ax.plot(epochs, losses, colour + "-.")
    acc_ax.plot(epochs, accs, colour + "-", label=name)
    if valid_accs is not None:
        acc_ax.plot(list(range(1, len(valid_accs) + 1)), valid_accs, colour + "o")

    return loss_ax, acc_ax


def visualise_training_data(losses, accuracies, epochs, name, show=False, valid_accs=None, title=None):
    # if epochs is None:
    #     epochs = list(range(len(losses)))
    losses = smooth(losses)
    accuracies = smooth(accuracies)
    plt.xlabel("Epoch")
    fig = plt.figure()
    if title is None:
        fig.suptitle(name)
    else:
        fig.suptitle(title)

    plot_loss_and_acc(losses, accuracies, epochs, name, valid_accs=valid_accs, fig=fig)
    plt.legend()

    if show:
        plt.show()
    path = loss_plot_path(name)
    plt.savefig(path)


def get_rolling_averages(losses: List[int], alph=0.95):
    """over the first 10 items, interp the alph value between (a-0.1) and a"""
    losses = copy.deepcopy(losses)
    avgs = [losses.pop(0)]
    while losses:
        next = losses.pop(0)

        alph_offset = max(0, 0.15 * (10 - len(avgs)))
        eff_alph = alph - alph_offset
        avg = avgs[-1] * eff_alph + next * (1-eff_alph)
        avgs.append(avg)
    return avgs


def remove_outliers(losses):
    """replaces outliers with the running mean before it"""
    means = get_rolling_averages(losses)
    avg = numpy.average(losses)
    dev = numpy.std(losses)
    cleans = []
    for i, loss in enumerate(losses):
        diff_roll, diff_mean = abs(loss - means[i]), abs(loss - avg)
        if diff_roll > 2 * dev or diff_mean > 3 * dev:
            """outlier"""
            if i == 0:
                cleans.append(avg)
            else:
                cleans.append(means[i])
        else:
            cleans.append(loss)
    return cleans


def plot_losses_from_lines(lines: List[str], show=False):
    """
        json lines (no commas between top level items)
        eg: {'loss': 4.89647998046875, 'learning_rate': 4.942921722850718e-05, 'epoch': 0.011415655429856505}

    """
    lines = [l for l in lines if "loss" in l]
    if '{' in lines[0]:  # jsonlines
        lines = [line.replace("'", '"') for line in lines]
        lines = [json.loads(line) for line in lines]
        epochs = [line["epoch"] for line in lines]
        losses = [line["loss"] for line in lines]
        accuracies=None
    else:
        has_tqdm = False
        off = 2 if has_tqdm else 0
        epochs = None
        losses = [float(l.split()[5 + off]) for l in lines]
        accuracies = [float(l.split()[11 + off]) for l in lines]
    print("epochs:", epochs)
    print("losses:", losses)
    visualise_training_data(losses, accuracies=accuracies, epochs=epochs, show=show, title="paste")


def plot_losses_from_paste_file(show=True):
    with open("paste_loss_print.txt") as f:
        j_lines = f.readlines()
        plot_losses_from_lines(j_lines, show=show)


if __name__ == "__main__":
    num_examples = 43738 if conf.max_examples == -1 else conf.max_examples
    print("num ex:", num_examples, "print:", conf.print_loss_every)
    compare(names=[["hde", "hde2", "hde3"], ["hde_gate", "hde_gate2", "hde_gate3"], ["hde_gate_glob"]], num_training_examples=num_examples, print_loss_every=conf.print_loss_every)