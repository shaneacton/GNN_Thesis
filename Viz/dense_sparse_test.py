import copy
import random
import time
from typing import Dict, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy import mean
from torch.nn import MultiheadAttention

from Code.GNNs.custom_gat import CustomGAT
from Code.HDE.Graph.edge import HDEEdge
from Code.HDE.Graph.graph import HDEGraph
from Code.HDE.Graph.node import HDENode
from Code.Training import dev
from Code.constants import TOKEN
from Viz.params_vs_acc import regress_data

NORMALISE = False
MEMORY = True

dim = 2
heads = 2
n = 15000

start_density = 0.005
end_density = 0.25
num_densities = 2

inner_repeats = 1
outter_repeats = 2
num_warmups = 1

densities = [start_density + i*(end_density-start_density)/(num_densities-1) for i in range(num_densities)]
densities = [end_density] * num_warmups + densities

dense = MultiheadAttention(dim, heads, batch_first=True).to(dev())
# sparse = GATConv(dim, dim//heads, heads, add_self_loops=False).to(dev())
sparse = CustomGAT(dim, dim, heads, add_self_loops=False).to(dev())

# print("dense:", num_params(dense), dense)
# print("sparse:", num_params(sparse), sparse)

x = torch.rand(n, dim).to(dev())
graph = None


def add_edges(density):
    global graph
    max_edges = n ** 2
    if density <= 0.5:
        num_edges = 0
        while num_edges < max_edges * density:
            edge = HDEEdge(random.randint(0, n - 1), random.randint(0, n - 1), safe_mode=True, graph=graph)
            if graph.has_edge(edge):
                continue
            graph.add_edge(edge)
            num_edges += 1
    else:
        num_edges = max_edges
        edges = [[True for _ in range(n)] for _ in range(n)]
        while num_edges > max_edges * density:
            i, j = random.randint(0, n - 1), random.randint(0, n - 1)
            if edges[i][j]:
                edges[i][j] = False
                num_edges -= 1
        for i in range(n):
            for j in range(n):
                if edges[i][j]:
                    edge = HDEEdge(i, j, safe_mode=True, graph=graph)
                    graph.add_edge(edge, safe_mode=True)


def get_graph(density):
    global graph
    global x
    graph = HDEGraph(None)
    x = torch.rand(n, dim).to(dev())

    for i in range(n):
        node = HDENode(TOKEN, text="")
        graph.add_node(node)

    add_edges(density)

    return graph


def get_mem_usage():
    # r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    # f = r - a  # free inside reserved
    return a/1048576


def measure_dense():
    global x
    x = x.view(1, n, dim)
    mask = graph.get_mask()
    # with torch.no_grad():
    start_time = time.perf_counter()
    outs = []
    for i in range(inner_repeats):
        d = dense(x, x, x, attn_mask=mask)
        outs.append(d)
    delta = time.perf_counter() - start_time
    if MEMORY:
        mem = get_mem_usage()
        return mem

    return delta/inner_repeats


def measure_sparse():
    global x
    x = x.view(n, dim)
    index = graph.edge_index()
    # with torch.no_grad():
    start_time = time.perf_counter()
    outs = []
    for i in range(inner_repeats):
        s = sparse(x, index)
        outs.append(s)
    delta = time.perf_counter() - start_time
    if MEMORY:
        mem = get_mem_usage()
        return mem
    return delta/inner_repeats


def measure_all():
    global graph
    s_times = []
    d_times = []
    for density in densities:
        d = 0
        s = 0
        for i in range(outter_repeats):
            graph = get_graph(density)
            d += measure_dense()
            s += measure_sparse()
        s_times.append(s/outter_repeats)
        d_times.append(d/outter_repeats)

        # print("density:", round(density, 3), "dense:", d_times[-1], "sparse:", s_times[-1])
    s_times, d_times = np.array(s_times[num_warmups:]), np.array(d_times[num_warmups:])
    if NORMALISE:
        s_times /= mean(s_times)
        d_times /= mean(d_times)
    return s_times, d_times


def plot_test(num_nodes, features, num_heads):
    global n, dim, heads
    n = num_nodes
    dim = features
    heads = num_heads
    s_times, d_times = measure_all()
    density_intersection, m, c = _get_density_intersection(s_times, d_times, return_mc=True)
    density_intersection = round(density_intersection, 3)
    time_intersection = density_intersection * m + c

    s_fac = s_times[-1] / s_times[0]
    d_fac = d_times[-1] / d_times[0]

    print("s fac:", s_fac, "d fac:", d_fac)

    plt.scatter(densities[num_warmups:], s_times, color="blue", label="sparse")
    m,c,r = regress_data(densities[num_warmups:], s_times)
    plt.plot([densities[num_warmups:][0], densities[num_warmups:][-1]],
             [densities[num_warmups:][0]*m + c, densities[num_warmups:][-1]*m +c])
    plt.scatter(densities[num_warmups:], d_times, color="red", label="dense")
    m, c, r = regress_data(densities[num_warmups:], d_times)
    plt.plot([densities[num_warmups:][0], densities[num_warmups:][-1]],
             [densities[num_warmups:][0] * m + c, densities[num_warmups:][-1] * m + c])

    title = "Memory usage scaling over Edge Density" if MEMORY else "Execution time scaling over Edge Density"
    title += "\nn=" + repr(n) + " f=" + repr(dim) + " heads:" + repr(heads)
    title += "\nDensity Intersection: " + repr(density_intersection)
    plt.title(title)
    plt.xlabel("Edge Density")
    ylabel = ('Normalised Memory Usage' if NORMALISE else 'Memory Usage(Mb)') if MEMORY else (
        'Normalised Time' if NORMALISE else 'Time(s)')
    plt.ylabel(ylabel)
    plt.legend(loc=2)

    plt.scatter([density_intersection], [time_intersection], marker="x", color="black")
    plt.show()


def _get_density_intersection(s_times, d_times, return_mc=False):
    sm, sc, sr = regress_data(densities[num_warmups:], s_times)
    dm, dc, dr = regress_data(densities[num_warmups:], d_times)

    x_intersect = (sc - dc) / (dm - sm)
    if return_mc:
        return x_intersect, sm, sc
    return x_intersect


def get_density_intersection(num_nodes, features, num_heads):
    global n, dim, heads, dense, sparse
    n = num_nodes
    dim = features
    heads = num_heads
    dense = MultiheadAttention(dim, num_heads, batch_first=True).to(dev())
    sparse = CustomGAT(dim, dim, num_heads, add_self_loops=False).to(dev())
    s_times, d_times = measure_all()
    return _get_density_intersection(s_times, d_times)


def plot_density_intersections(fixed_values: Dict, loop_values:List):
    """
    fixed_values: {"num_nodes": n, "features": f, "num_heads": h}
    keep one value out eg: {"num_nodes": n, "features": f, "num_heads": None}
    """
    loop_key = None
    intersections = []
    for val in loop_values:
        f_vals = copy.deepcopy(fixed_values)
        for key in f_vals.keys():
            if f_vals[key] is None:
                f_vals[key] = val
                loop_key = key
        density_intersection = get_density_intersection(**f_vals)
        intersections.append(density_intersection)
        print(loop_key + ": " + repr(val), "density intersection:", density_intersection)
    plt.scatter(loop_values, intersections)
    # m,c,r = regress_data(loop_values, intersections)
    # plt.plot([loop_values[0], loop_values[-1]], [loop_values[0]*m + c, loop_values[-1]*m +c])

    title = "Density intersection VS " + loop_key + "\n"
    for key in fixed_values.keys():
        if key == loop_key:
            continue
        title += key + ": " + repr(fixed_values[key]) + " "

    plt.title(title)
    plt.xlabel(loop_key)
    plt.ylabel("Density intersection")
    plt.show()


if __name__ == "__main__":
    plt.rcParams.update({'font.size': 15})

    plot_density_intersections({"num_nodes": 100, "features": 100, "num_heads": None}, [1, 10, 20, 50, 100])
    plot_density_intersections({"num_nodes": 100, "features": None, "num_heads": 10}, [10, 20, 30, 50, 100, 300, 500, 750, 1000])
    plot_density_intersections({"num_nodes": None, "features": 10, "num_heads": 10}, [20, 30, 50, 100, 200, 300, 450, 600, 800, 1000, 1250, 1500])
    # plot_test(n, dim, heads)






