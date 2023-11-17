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
# 2: https://github.com/erikbern/ann-benchmarks/blob/main/ann_benchmarks/plotting/utils.py  # noqa: E501
# 3: https://github.com/erikbern/ann-benchmarks/blob/main/ann_benchmarks/plotting/metrics.py  # noqa: E501
# Licence: https://github.com/erikbern/ann-benchmarks/blob/main/LICENSE

import argparse
import itertools
import os
import sys
from collections import OrderedDict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.use("Agg")

metrics = {
    "k-nn": {
        "description": "Recall",
        "worst": float("-inf"),
        "lim": [0.0, 1.03],
    },
    "throughput": {
        "description": "Queries per second (1/s)",
        "worst": float("-inf"),
    },
    "latency": {
        "description": "Search Latency (s)",
        "worst": float("inf"),
    },
}


def positive_int(input_str: str) -> int:
    try:
        i = int(input_str)
        if i < 1:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{input_str} is not a positive integer"
        )

    return i


def generate_n_colors(n):
    vs = np.linspace(0.3, 0.9, 7)
    colors = [(0.9, 0.4, 0.4, 1.0)]

    def euclidean(a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b))

    while len(colors) < n:
        new_color = max(
            itertools.product(vs, vs, vs),
            key=lambda a: min(euclidean(a, b) for b in colors),
        )
        colors.append(new_color + (1.0,))
    return colors


def create_linestyles(unique_algorithms):
    colors = dict(
        zip(unique_algorithms, generate_n_colors(len(unique_algorithms)))
    )
    linestyles = dict(
        (algo, ["--", "-.", "-", ":"][i % 4])
        for i, algo in enumerate(unique_algorithms)
    )
    markerstyles = dict(
        (algo, ["+", "<", "o", "*", "x"][i % 5])
        for i, algo in enumerate(unique_algorithms)
    )
    faded = dict(
        (algo, (r, g, b, 0.3)) for algo, (r, g, b, a) in colors.items()
    )
    return dict(
        (
            algo,
            (colors[algo], faded[algo], linestyles[algo], markerstyles[algo]),
        )
        for algo in unique_algorithms
    )


def create_plot_search(
    all_data,
    x_scale,
    y_scale,
    fn_out,
    linestyles,
    dataset,
    k,
    batch_size,
    mode,
):
    xn = "k-nn"
    xm, ym = (metrics[xn], metrics[mode])
    # Now generate each plot
    handles = []
    labels = []
    plt.figure(figsize=(12, 9))

    # Sorting by mean y-value helps aligning plots with labels
    def mean_y(algo):
        points = np.array(all_data[algo], dtype=object)
        return -np.log(np.array(points[:, 3], dtype=np.float32)).mean()

    # Find range for logit x-scale
    min_x, max_x = 1, 0
    for algo in sorted(all_data.keys(), key=mean_y):
        points = np.array(all_data[algo], dtype=object)
        xs = points[:, 2]
        ys = points[:, 3]
        min_x = min([min_x] + [x for x in xs if x > 0])
        max_x = max([max_x] + [x for x in xs if x < 1])
        color, faded, linestyle, marker = linestyles[algo]
        (handle,) = plt.plot(
            xs,
            ys,
            "-",
            label=algo,
            color=color,
            ms=7,
            mew=3,
            lw=3,
            marker=marker,
        )
        handles.append(handle)

        labels.append(algo)

    ax = plt.gca()
    ax.set_ylabel(ym["description"])
    ax.set_xlabel("Recall")
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
    ax.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        prop={"size": 9},
    )
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


def create_plot_build(
    build_results, search_results, linestyles, fn_out, dataset
):

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
        points = np.array(search_results[algo], dtype=object)
        return -np.log(np.array(points[:, 3], dtype=np.float32)).mean()

    for pos, algo in enumerate(sorted(search_results.keys(), key=mean_y)):
        points = np.array(search_results[algo], dtype=object)
        xs = points[:, 2]
        ys = points[:, 3]
        ls = points[:, 0]
        idxs = points[:, 1]
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

    index = ["@85% Recall", "@90% Recall", "@95% Recall"]

    df = pd.DataFrame(data, index=index)
    plt.figure(figsize=(12, 9))
    ax = df.plot.bar(rot=0, color=colors)
    fig = ax.get_figure()
    print(f"writing build output to {fn_out}")
    plt.title("Build Time for Highest QPS")
    plt.suptitle(f"{dataset}")
    plt.ylabel("Build Time (s)")
    fig.savefig(fn_out)


def load_lines(results_path, result_files, method, index_key, mode):
    results = dict()

    for result_filename in result_files:
        if result_filename.endswith(".csv"):
            with open(os.path.join(results_path, result_filename), "r") as f:
                lines = f.readlines()
                lines = lines[:-1] if lines[-1] == "\n" else lines

                if method == "build":
                    key_idx = [2]
                elif method == "search":
                    y_idx = 3 if mode == "throughput" else 4
                    key_idx = [2, y_idx]

                for line in lines[1:]:
                    split_lines = line.split(",")

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


def load_all_results(
    dataset_path,
    algorithms,
    groups,
    algo_groups,
    k,
    batch_size,
    method,
    index_key,
    raw,
    mode,
):
    results_path = os.path.join(dataset_path, "result", method)
    result_files = os.listdir(results_path)
    if method == "build":
        result_files = [
            result_file
            for result_file in result_files
            if ".csv" in result_file
        ]
    elif method == "search":
        if raw:
            suffix = "_raw"
        else:
            suffix = f"_{mode}"
        result_files = [
            result_file
            for result_file in result_files
            if f"{suffix}.csv" in result_file
        ]
    if method == "search":
        result_files = [
            result_filename
            for result_filename in result_files
            if f"{k}-{batch_size}" in result_filename
        ]
        algo_group_files = [
            result_filename.split("-")[0] for result_filename in result_files
        ]
    else:
        algo_group_files = [
            result_filename for result_filename in result_files
        ]

    for i in range(len(algo_group_files)):
        algo_group = algo_group_files[i].replace(".csv", "").split("_")
        algo_group_files[i] = ("_".join(algo_group[:-1]), algo_group[-1])
    algo_group_files = list(zip(*algo_group_files))

    if len(algorithms) > 0:
        final_results = [
            result_files[i]
            for i in range(len(result_files))
            if (algo_group_files[0][i] in algorithms)
            and (algo_group_files[1][i] in groups)
        ]
    else:
        final_results = [
            result_files[i]
            for i in range(len(result_files))
            if (algo_group_files[1][i] in groups)
        ]

    if len(algo_groups) > 0:
        split_algo_groups = [
            algo_group.split(".") for algo_group in algo_groups
        ]
        split_algo_groups = list(zip(*split_algo_groups))
        final_algo_groups = [
            result_files[i]
            for i in range(len(result_files))
            if (algo_group_files[0][i] in split_algo_groups[0])
            and (algo_group_files[1][i] in split_algo_groups[1])
        ]
        final_results = final_results + final_algo_groups
        final_results = set(final_results)

    results = load_lines(results_path, final_results, method, index_key, mode)

    return results


def main():
    call_path = os.getcwd()
    if "RAPIDS_DATASET_ROOT_DIR" in os.environ:
        default_dataset_path = os.getenv("RAPIDS_DATASET_ROOT_DIR")
    else:
        default_dataset_path = os.path.join(call_path, "datasets/")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset", help="dataset to plot", default="glove-100-inner"
    )
    parser.add_argument(
        "--dataset-path",
        help="path to dataset folder",
        default=default_dataset_path,
    )
    parser.add_argument(
        "--output-filepath",
        help="directory for PNG to be saved",
        default=os.getcwd(),
    )
    parser.add_argument(
        "--algorithms",
        help="plot only comma separated list of named \
              algorithms. If parameters `groups` and `algo-groups \
              are both undefined, then group `base` is plot by default",
        default=None,
    )
    parser.add_argument(
        "--groups",
        help="plot only comma separated groups of parameters",
        default="base",
    )
    parser.add_argument(
        "--algo-groups",
        "--algo-groups",
        help='add comma separated <algorithm>.<group> to plot. \
              Example usage: "--algo-groups=raft_cagra.large,hnswlib.large"',
    )
    parser.add_argument(
        "-k",
        "--count",
        default=10,
        type=positive_int,
        help="the number of nearest neighbors to search for",
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        default=10000,
        type=positive_int,
        help="number of query vectors to use in each query trial",
    )
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--search", action="store_true")
    parser.add_argument(
        "--x-scale",
        help="Scale to use when drawing the X-axis. \
              Typically linear, logit or a2",
        default="linear",
    )
    parser.add_argument(
        "--y-scale",
        help="Scale to use when drawing the Y-axis",
        choices=["linear", "log", "symlog", "logit"],
        default="linear",
    )
    parser.add_argument(
        "--mode",
        help="metric whose Pareto frontier is used on the y-axis",
        choices=["throughput", "latency"],
        default="throughput",
    )
    parser.add_argument(
        "--raw",
        help="Show raw results (not just Pareto frontier) of metric arg",
        action="store_true",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    if args.algorithms:
        algorithms = args.algorithms.split(",")
    else:
        algorithms = []
    groups = args.groups.split(",")
    if args.algo_groups:
        algo_groups = args.algo_groups.split(",")
    else:
        algo_groups = []
    k = args.count
    batch_size = args.batch_size
    if not args.build and not args.search:
        build = True
        search = True
    else:
        build = args.build
        search = args.search

    search_output_filepath = os.path.join(
        args.output_filepath,
        f"search-{args.dataset}-k{k}-batch_size{batch_size}.png",
    )
    build_output_filepath = os.path.join(
        args.output_filepath,
        f"build-{args.dataset}.png",
    )

    search_results = load_all_results(
        os.path.join(args.dataset_path, args.dataset),
        algorithms,
        groups,
        algo_groups,
        k,
        batch_size,
        "search",
        "algo",
        args.raw,
        args.mode,
    )
    linestyles = create_linestyles(sorted(search_results.keys()))
    if search:
        create_plot_search(
            search_results,
            args.x_scale,
            args.y_scale,
            search_output_filepath,
            linestyles,
            args.dataset,
            k,
            batch_size,
            args.mode,
        )
    if build:
        build_results = load_all_results(
            os.path.join(args.dataset_path, args.dataset),
            algorithms,
            groups,
            algo_groups,
            k,
            batch_size,
            "build",
            "index",
            args.raw,
            args.mode,
        )
        create_plot_build(
            build_results,
            search_results,
            linestyles,
            build_output_filepath,
            args.dataset,
        )


if __name__ == "__main__":
    main()
