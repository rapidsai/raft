#
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script is inspired by 
# 1: https://github.com/erikbern/ann-benchmarks/blob/main/plot.py
# 2: https://github.com/erikbern/ann-benchmarks/blob/main/ann_benchmarks/plotting/utils.py
# 3: https://github.com/erikbern/ann-benchmarks/blob/main/ann_benchmarks/plotting/metrics.py
# Licence: https://github.com/erikbern/ann-benchmarks/blob/main/LICENSE

import matplotlib as mpl

mpl.use("Agg")  # noqa
import argparse
from collections import OrderedDict
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os



metrics = {
    "k-nn": {
        "description": "Recall",
        "worst": float("-inf"),
        "lim": [0.0, 1.03],
    },
    "qps": {
        "description": "Queries per second (1/s)",
        "worst": float("-inf"),
    }
}

def positive_int(input_str: str) -> int:
    try:
        i = int(input_str)
        if i < 1:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(f"{input_str} is not a positive integer")

    return i


def generate_n_colors(n):
    vs = np.linspace(0.3, 0.9, 7)
    colors = [(0.9, 0.4, 0.4, 1.0)]

    def euclidean(a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b))

    while len(colors) < n:
        new_color = max(itertools.product(vs, vs, vs), key=lambda a: min(euclidean(a, b) for b in colors))
        colors.append(new_color + (1.0,))
    return colors


def create_linestyles(unique_algorithms):
    colors = dict(zip(unique_algorithms, generate_n_colors(len(unique_algorithms))))
    linestyles = dict((algo, ["--", "-.", "-", ":"][i % 4]) for i, algo in enumerate(unique_algorithms))
    markerstyles = dict((algo, ["+", "<", "o", "*", "x"][i % 5]) for i, algo in enumerate(unique_algorithms))
    faded = dict((algo, (r, g, b, 0.3)) for algo, (r, g, b, a) in colors.items())
    return dict((algo, (colors[algo], faded[algo], linestyles[algo], markerstyles[algo])) for algo in unique_algorithms)


def get_up_down(metric):
    if metric["worst"] == float("inf"):
        return "down"
    return "up"


def get_left_right(metric):
    if metric["worst"] == float("inf"):
        return "left"
    return "right"


def create_pointset(data, xn, yn):
    xm, ym = (metrics[xn], metrics[yn])
    rev_y = -1 if ym["worst"] < 0 else 1
    rev_x = -1 if xm["worst"] < 0 else 1
    data.sort(key=lambda t: (rev_y * t[-1], rev_x * t[-2]))

    axs, ays, als, aidxs = [], [], [], []
    # Generate Pareto frontier
    xs, ys, ls, idxs = [], [], [], []
    last_x = xm["worst"]
    comparator = (lambda xv, lx: xv > lx) if last_x < 0 else (lambda xv, lx: xv < lx)
    for algo_name, index_name, xv, yv in data:
        if not xv or not yv:
            continue
        axs.append(xv)
        ays.append(yv)
        als.append(algo_name)
        aidxs.append(algo_name)
        if comparator(xv, last_x):
            last_x = xv
            xs.append(xv)
            ys.append(yv)
            ls.append(algo_name)
            idxs.append(index_name)
    return xs, ys, ls, idxs, axs, ays, als, aidxs


def create_plot_search(all_data, raw, x_scale, y_scale, fn_out, linestyles,
                dataset, k, batch_size):
    xn = "k-nn"
    yn = "qps"
    xm, ym = (metrics[xn], metrics[yn])
    # Now generate each plot
    handles = []
    labels = []
    plt.figure(figsize=(12, 9))

    # Sorting by mean y-value helps aligning plots with labels
    def mean_y(algo):
        xs, ys, ls, idxs, axs, ays, als, aidxs = create_pointset(all_data[algo], xn, yn)
        return -np.log(np.array(ys)).mean()

    # Find range for logit x-scale
    min_x, max_x = 1, 0
    for algo in sorted(all_data.keys(), key=mean_y):
        xs, ys, ls, idxs, axs, ays, als, aidxs = create_pointset(all_data[algo], xn, yn)
        min_x = min([min_x] + [x for x in xs if x > 0])
        max_x = max([max_x] + [x for x in xs if x < 1])
        color, faded, linestyle, marker = linestyles[algo]
        (handle,) = plt.plot(
            xs, ys, "-", label=algo, color=color, ms=7, mew=3, lw=3, marker=marker
        )
        handles.append(handle)
        if raw:
            (handle2,) = plt.plot(
                axs, ays, "-", label=algo, color=faded, ms=5, mew=2, lw=2, marker=marker
            )
        labels.append(algo)

    ax = plt.gca()
    ax.set_ylabel(ym["description"])
    ax.set_xlabel(xm["description"])
    # Custom scales of the type --x-scale a3
    if x_scale[0] == "a":
        alpha = float(x_scale[1:])

        def fun(x):
            return 1 - (1 - x) ** (1 / alpha)

        def inv_fun(x):
            return 1 - (1 - x) ** alpha

        ax.set_xscale("function", functions=(fun, inv_fun))
        if alpha <= 3:
            ticks = [inv_fun(x) for x in np.arange(0, 1.2, 0.2)]
            plt.xticks(ticks)
        if alpha > 3:
            from matplotlib import ticker

            ax.xaxis.set_major_formatter(ticker.LogitFormatter())
            # plt.xticks(ticker.LogitLocator().tick_values(min_x, max_x))
            plt.xticks([0, 1 / 2, 1 - 1e-1, 1 - 1e-2, 1 - 1e-3, 1 - 1e-4, 1])
    # Other x-scales
    else:
        ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    ax.set_title(f"{dataset} k={k} batch_size={batch_size}")
    plt.gca().get_position()
    # plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), prop={"size": 9})
    plt.grid(visible=True, which="major", color="0.65", linestyle="-")
    plt.setp(ax.get_xminorticklabels(), visible=True)

    # Logit scale has to be a subset of (0,1)
    if "lim" in xm and x_scale != "logit":
        x0, x1 = xm["lim"]
        plt.xlim(max(x0, 0), min(x1, 1))
    elif x_scale == "logit":
        plt.xlim(min_x, max_x)
    if "lim" in ym:
        plt.ylim(ym["lim"])

    # Workaround for bug https://github.com/matplotlib/matplotlib/issues/6789
    ax.spines["bottom"]._adjust_location()

    print(f"writing search output to {fn_out}")
    plt.savefig(fn_out, bbox_inches="tight")
    plt.close()


def create_plot_build(build_results, search_results, linestyles, fn_out,
                      dataset, k, batch_size):
    xn = "k-nn"
    yn = "qps"

    qps_85 = [-1] * len(linestyles)
    bt_85 = [0] * len(linestyles)
    i_85 = [-1] * len(linestyles)

    qps_90 = [-1] * len(linestyles)
    bt_90 = [0] * len(linestyles)
    i_90 = [-1] * len(linestyles)

    qps_95 = [-1] * len(linestyles)
    bt_95 = [0] * len(linestyles)
    i_95 = [-1] * len(linestyles)

    data = OrderedDict()
    colors = OrderedDict()

    # Sorting by mean y-value helps aligning plots with labels
    def mean_y(algo):
        xs, ys, ls, idxs, axs, ays, als, aidxs = create_pointset(search_results[algo], xn, yn)
        return -np.log(np.array(ys)).mean()

    for pos, algo in enumerate(sorted(search_results.keys(), key=mean_y)):
        xs, ys, ls, idxs, axs, ays, als, aidxs = create_pointset(search_results[algo], xn, yn)
        # x is recall, y is qps, ls is algo_name, idxs is index_name
        for i in range(len(xs)):
            if xs[i] >= 0.85 and xs[i] < 0.9 and ys[i] > qps_85[pos]:
                qps_85[pos] = ys[i]
                bt_85[pos] = build_results[(ls[i], idxs[i])][0][2]
                i_85[pos] = idxs[i]
            elif xs[i] >= 0.9 and xs[i] < 0.95 and ys[i] > qps_90[pos]:
                qps_90[pos] = ys[i]
                bt_90[pos] = build_results[(ls[i], idxs[i])][0][2]
                i_90[pos] = idxs[i]
            elif xs[i] >= 0.95 and ys[i] > qps_95[pos]:
                qps_95[pos] = ys[i]
                bt_95[pos] = build_results[(ls[i], idxs[i])][0][2]
                i_95[pos] = idxs[i]
        data[algo] = [bt_85[pos], bt_90[pos], bt_95[pos]]
        colors[algo] = linestyles[algo][0]
    
    index = ['@85% Recall', '@90% Recall', '@95% Recall']
    
    df = pd.DataFrame(data, index=index)
    plt.figure(figsize=(12, 9))
    ax = df.plot.bar(rot=0, color=colors)
    fig = ax.get_figure()
    print(f"writing build output to {fn_out}")
    plt.title("Build Time for Highest QPS")
    plt.suptitle(f"{dataset} k={k} batch_size={batch_size}")
    plt.ylabel("Build Time (s)")
    fig.savefig(fn_out)


def load_lines(results_path, result_files, method, index_key):
    results = dict()

    for result_filename in result_files:
        if result_filename.endswith('.csv'):
            with open(os.path.join(results_path, result_filename), 'r') as f:
                lines = f.readlines()
                lines = lines[:-1] if lines[-1] == "\n" else lines
                
                if method == "build":
                    key_idx = [2]
                elif method == "search":
                    key_idx = [2, 3]

                for line in lines[1:]:
                    split_lines = line.split(',')

                    algo_name = split_lines[0]
                    index_name = split_lines[1]

                    if index_key == "algo":
                        dict_key = algo_name
                    elif index_key == "index":
                        dict_key = (algo_name, index_name)
                    if dict_key not in results:
                        results[dict_key] = []
                    to_add = [algo_name, index_name]
                    for key_i in key_idx:
                        to_add.append(float(split_lines[key_i]))
                    results[dict_key].append(to_add)

    return results


def load_all_results(dataset_path, algorithms, k, batch_size, method, index_key):
    results_path = os.path.join(dataset_path, "result", method)
    result_files = os.listdir(results_path)
    result_files = [result_filename for result_filename in result_files \
                    if f"{k}-{batch_size}" in result_filename]
    if len(algorithms) > 0:
        result_files = [result_filename for result_filename in result_files if \
                        result_filename.split('-')[0] in algorithms]

    results = load_lines(results_path, result_files, method, index_key)

    return results


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", help="dataset to download",
                        default="glove-100-inner")
    parser.add_argument("--dataset-path", help="path to dataset folder",
                        default=os.path.join(os.getenv("RAFT_HOME"), 
                                             "bench", "ann", "data"))
    parser.add_argument("--output-filepath",
                        help="directory for PNG to be saved",
                        default=os.getcwd())
    parser.add_argument("--algorithms",
                        help="plot only comma separated list of named \
                              algorithms",
                        default=None)
    parser.add_argument(
        "-k", "--count", default=10, type=positive_int, help="the number of nearest neighbors to search for"
    )
    parser.add_argument(
        "-bs", "--batch-size", default=10000, type=positive_int, help="number of query vectors to use in each query trial"
    )
    parser.add_argument(
        "--build",
        action="store_true"
    )
    parser.add_argument(
        "--search",
        action="store_true"
    )
    parser.add_argument(
        "--x-scale",
        help="Scale to use when drawing the X-axis. \
              Typically linear, logit or a2", 
        default="linear"
    )
    parser.add_argument(
        "--y-scale",
        help="Scale to use when drawing the Y-axis",
        choices=["linear", "log", "symlog", "logit"],
        default="linear",
    )
    parser.add_argument(
        "--raw", help="Show raw results (not just Pareto frontier) in faded colours", action="store_true"
    )

    args = parser.parse_args()

    if args.algorithms:
        algorithms = args.algorithms.split(',')
    else:
        algorithms = []
    k = args.count
    batch_size = args.batch_size
    if not args.build and not args.search:
        build = True
        search = True
    else:
        build = args.build
        search = args.search

    search_output_filepath = os.path.join(args.output_filepath, f"search-{args.dataset}-k{k}-batch_size{batch_size}.png")
    build_output_filepath = os.path.join(args.output_filepath, f"build-{args.dataset}-k{k}-batch_size{batch_size}.png")

    search_results = load_all_results(
                        os.path.join(args.dataset_path, args.dataset),
                        algorithms, k, batch_size, "search", "algo")
    linestyles = create_linestyles(sorted(search_results.keys()))
    if search:
        create_plot_search(search_results, args.raw, args.x_scale, args.y_scale, 
                           search_output_filepath, linestyles, args.dataset, k, batch_size)
    if build:
        build_results = load_all_results(
            os.path.join(args.dataset_path, args.dataset),
            algorithms, k, batch_size, "build", "index")
        create_plot_build(build_results, search_results, linestyles, build_output_filepath,
                          args.dataset, k, batch_size)


if __name__ == "__main__":
    main()
