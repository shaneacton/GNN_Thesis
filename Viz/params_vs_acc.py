# GROUP_NAME = "MLP-Base"

# # base_0, base_0_switch, base_0_sage, *base_0_gru, base_0_NoShare
# ACCURACIES = [60, 58.9, 59.6, 57.6, 62]
# NUM_PARAMS = [3.2, 3.7, 3.3, 8.3, 6.1]

# # base0 new, new no gate, new tuf, new tuf no gate, newnew no gate, newnew, new sage, *new sage no gate*,
# # base0 linear2, newnew switch, newnew switch no gate, realsage, sagecore gate, base0 edge, realsage nogate,
# # new gat, new dweight, new no edge, new grus, new faith
# ACCURACIES = [60.5, 52.7, 61.5, 64.1, 58.2, 60.3, 61.9, *, 58.9, 58.0, 58.6, 58.0, 57.4, 59.6, 64.7, 62, 58.1, 57.6, 54.4]
# NUM_PARAMS = [3.2,  2.9,  3.6,  3.3,  3.0,  3.3,  3.4,  *, 3.7,  3.8,  3.5,  3.3,  3.1,  3.3,  3.2, 1.4, 3.2, 8.3, 10.5]

# ACCURACIES = [60.5, 52.7, 61.5, 64.1, 58.2, 60.3, 61.9, 58.9, 58.0, 58.6, 58.0, 57.4, 59.6, 64.7, 62, 58.1, 57.6, 54.4]
# NUM_PARAMS = [3.2,  2.9,  3.6,  3.3,  3.0,  3.3,  3.4,  3.7,  3.8,  3.5,  3.3,  3.1,  3.3,  3.2,  1.4, 3.2, 8.3, 10.5]
from typing import List

GROUP_NAME = "GAT-Base"

# base2 trans, trans no gate, base2, sdp, sdp trans no tuf, sdp trans no gate, trans linear, trans linear no gate, 2 linear,
ACCURACIES = [64,   60.4, 62.9, 64.7, 61.4, 66,  60.2, 62.5, 58.5]
NUM_PARAMS = [10.8, 8.4,  7.5,  13.2, 10.0, 10.8, 6.5, 4.1, 3.3]

# trans_no_switch, trans_share, trans_sdp_all_edges, trans_sdp_structure, trans_sdp, trans_sdp_graph_vtype,
# trans_sharetuf, trans_sharegate, trans_sharetufgate, trans_sharegnntuf, trans_sharegnn, trans_sdp_graph_vtype_ktype
ACCURACIES += [64.1, 63.6, 64.4, 60.2, 61.5, 65.1, 63.3, 62.6, 61.1, 64.3, 63.6, 64.6]
NUM_PARAMS += [9.4,  5.0,  13.2, 13.2, 13.2, 13.2, 7.9,  8.6,  5.8,  7.2,  10.1, 13.3]

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
    plt.rcParams.update({'font.size': 15})

    plt.scatter(params, accs, color="blue")

    plt.title(GROUP_NAME + " Parameter Accuracy Correlation. n=" + repr(len(accs)))
    plt.xlabel("Num Model Parameters (Millions)")
    plt.ylabel('Accuracy %')

    m, c, r = regress_data(params, accs)
    params = np.array(params)
    plt.plot(params, params * m + c, color="red", label="R = " + repr(r))


    plt.legend(loc=4)
    plt.show()


if __name__ == "__main__":
    plot(ACCURACIES, NUM_PARAMS)

