import copy
import json
from typing import List
import matplotlib.pyplot as plt
import numpy


def plot_losses(epochs, losses):
    losses = remove_outliers(losses)
    losses = get_rolling_averages(losses)
    plt.plot(epochs, losses)
    plt.show()


def get_rolling_averages(losses: List[int], alph=0.95):
    losses = copy.deepcopy(losses)
    avgs = [losses.pop(0)]
    while losses:
        next = losses.pop(0)
        avg = avgs[-1] * alph + next * (1-alph)
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


def plot_losses_from_lines(lines: List[str]):
    """
        json lines (no commas between top level items)
        eg: {'loss': 4.89647998046875, 'learning_rate': 4.942921722850718e-05, 'epoch': 0.011415655429856505}

    """
    lines = [line.replace("'", '"') for line in lines]
    lines = [json.loads(line) for line in lines]
    epochs = [line["epoch"] for line in lines]
    losses = [line["loss"] for line in lines]
    plot_losses(epochs, losses)


def plot_losses_from_paste_file():
    with open("paste_loss_print.txt") as f:
        j_lines = f.readlines()
        plot_losses_from_lines(j_lines)


if __name__ == "__main__":
    plot_losses_from_paste_file()