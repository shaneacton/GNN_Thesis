# GROUP_NAME = "Base0"
# ACCURACIES = [60, 58.9, 59.6, 57.6, 62]  # base_0, base_0_switch, base_0_sage, *base_0_gru, base_0_NoShare
# NUM_PARAMS = [3.2, 3.7, 3.3, 8.3, 6.1]

# # base1, base1_SDP, base1_sage, base1_linear, base1_nogate, base1_noTuf, base1_noSS, base1_share
from typing import List

GROUP_NAME = "Base1"
ACCURACIES = [64, 64.7, 57.3, 60.1, 60.3, 62.9, 65, 63.6]
NUM_PARAMS = [10.8, 13.2, 11.6, 10.8, 8.4, 7.5, 6.1, 5.0]

import matplotlib.pyplot as plt
import numpy as np


def regress_data(x: List, y: List):
    if x is List:
        x, y = np.array(x), np.array(y)
    r = round(np.corrcoef(x, y)[1, 0], 3)
    X = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(X, y)[0]

    return m, c, r


def plot(accs, params):
    plt.scatter(params, accs, color="blue")

    plt.title(GROUP_NAME + " Parameter Accuracy Correlation. n=" + repr(len(accs)))
    plt.xlabel("Num Model Parameters (Millions)")
    plt.ylabel('Accuracy %')

    m, c, r = regress_data(params, accs)
    params = np.array(params)
    plt.plot(params, params * m + c, color="red", label="R = " + repr(r))

    plt.legend(loc=2)
    plt.show()


if __name__ == "__main__":
    plot(ACCURACIES, NUM_PARAMS)

