import copy
import json
from typing import List
import matplotlib.pyplot as plt
import numpy


def visualise_training_data(losses, epochs=None, accuracies=None, show=False, save_path=None, valid_accs=None):
    if epochs is None:
        epochs = list(range(len(losses)))
    losses = remove_outliers(losses)
    losses = get_rolling_averages(losses)
    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    ax1.plot(epochs, losses)
    ax1.set_ylabel('loss', color='b')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')

    if accuracies is not None:
        accuracies = remove_outliers(accuracies)
        accuracies = get_rolling_averages(accuracies)

        ax2 = ax1.twinx()
        ax2.plot(epochs, accuracies, 'r-')
        ax2.set_ylabel('accuracy', color='r')
        for tl in ax2.get_yticklabels():
            tl.set_color('r')

        if valid_accs is not None:
            ax2.plot(list(range(1, len(valid_accs) + 1)), valid_accs, "ro")

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
    visualise_training_data(losses, accuracies=accuracies, epochs=epochs, show=show)


def plot_losses_from_paste_file(show=True):
    with open("paste_loss_print.txt") as f:
        j_lines = f.readlines()
        plot_losses_from_lines(j_lines, show=show)


if __name__ == "__main__":
    plot_losses_from_paste_file()