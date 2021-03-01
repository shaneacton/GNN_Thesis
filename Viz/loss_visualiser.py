import copy
import json
from typing import List
import matplotlib.pyplot as plt
import numpy

import os
import sys
from os.path import join

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
dir_path_1 = join(dir_path_1, "GNN_Thesis")
sys.path.append(dir_path_1)
sys.path.append(os.path.join(dir_path_1, 'Code'))
sys.path.append(os.path.join(dir_path_1, 'conf'))

from Config import load_config
from Config.config import conf


def compare(save_paths: List[str]=None, names=None, num_training_examples=43700, show=True, print_loss_every=None):
    from Code.Main.scheduler import CHECKPOINT_FOLDER
    from Code.Training.Utils.training_utils import get_training_results

    if save_paths is None:
        save_paths = [join(CHECKPOINT_FOLDER, n) for n in names]

    loss_ax, acc_ax = None, None
    colours = ["g", "b", "r"]
    for i, save_path in enumerate(save_paths):
        data = get_training_results(save_path)
        losses = smooth(data["losses"])
        # print("got losses:", losses)
        # print("from:", save_path)
        train_accs = smooth(data["train_accs"])
        valid_accs = data["valid_accs"]
        epochs = get_continuous_epochs(losses, num_training_examples, print_loss_every=print_loss_every)
        loss_ax, acc_ax = plot_loss_and_acc(losses, train_accs, save_path=save_path, valid_accs=valid_accs,
                                            epochs=epochs, loss_ax=loss_ax, acc_ax=acc_ax, colour=colours[i])

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


def get_continuous_epochs(losses, num_training_examples, print_loss_every=None):
    if print_loss_every is None:
        train_conf = load_config("standard_train")
        print_loss_every = train_conf["print_loss_every"]
    num_prints = len(losses)
    num_trained_examples = num_prints * print_loss_every
    num_epochs = num_trained_examples / num_training_examples
    epochs = [num_epochs * i / len(losses) for i in range(len(losses))]
    return epochs


def plot_loss_and_acc(losses, accs, epochs, name=None, save_path=None, valid_accs=None, fig=None, loss_ax=None, acc_ax=None, colour="g"):
    if fig is None and loss_ax is None:
        fig = plt.figure()

    if name is None:
        name = save_path_to_name(save_path)

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


def visualise_training_data(losses, accuracies, epochs, show=False, save_path=None, valid_accs=None, title=None):
    # if epochs is None:
    #     epochs = list(range(len(losses)))
    losses = smooth(losses)
    accuracies = smooth(accuracies)
    plt.xlabel("Epoch")
    fig = plt.figure()
    if title is None:
        fig.suptitle(save_path.split("/")[-1])
    else:
        fig.suptitle(title)

    plot_loss_and_acc(losses, accuracies, epochs, valid_accs=valid_accs, save_path=save_path, fig=fig)
    plt.legend()

    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)


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
    compare(names=["hde", "hde_no_pool", "hde_pool_5"], num_training_examples=num_examples, print_loss_every=conf.print_loss_every)