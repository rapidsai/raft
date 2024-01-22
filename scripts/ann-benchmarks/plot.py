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
import itertools
import matplotlib.pyplot as plt
import numpy as np
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


def get_plot_label(xm, ym):
    template = "%(xlabel)s-%(ylabel)s tradeoff - %(updown)s and" " to the %(leftright)s is better"
    return template % {
        "xlabel": xm["description"],
        "ylabel": ym["description"],
        "updown": get_up_down(ym),
        "leftright": get_left_right(xm),
    }


def create_pointset(data, xn, yn):
    xm, ym = (metrics[xn], metrics[yn])
    rev_y = -1 if ym["worst"] < 0 else 1
    rev_x = -1 if xm["worst"] < 0 else 1
    data.sort(key=lambda t: (rev_y * t[-1], rev_x * t[-2]))

    axs, ays, als = [], [], []
    # Generate Pareto frontier
    xs, ys, ls = [], [], []
    last_x = xm["worst"]
    comparator = (lambda xv, lx: xv > lx) if last_x < 0 else (lambda xv, lx: xv < lx)
    for algo_name, xv, yv in data:
        if not xv or not yv:
            continue
        axs.append(xv)
        ays.append(yv)
        als.append(algo_name)
        if comparator(xv, last_x):
            last_x = xv
            xs.append(xv)
            ys.append(yv)
            ls.append(algo_name)
    return xs, ys, ls, axs, ays, als


def create_plot(all_data, raw, x_scale, y_scale, fn_out, linestyles):
    xn = "k-nn"
    yn = "qps"
    xm, ym = (metrics[xn], metrics[yn])
    # Now generate each plot
    handles = []
    labels = []
    plt.figure(figsize=(12, 9))

    # Sorting by mean y-value helps aligning plots with labels
    def mean_y(algo):
        xs, ys, ls, axs, ays, als = create_pointset(all_data[algo], xn, yn)
        return -np.log(np.array(ys)).mean()

    # Find range for logit x-scale
    min_x, max_x = 1, 0
    for algo in sorted(all_data.keys(), key=mean_y):
        xs, ys, ls, axs, ays, als = create_pointset(all_data[algo], xn, yn)
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
    ax.set_title(get_plot_label(xm, ym))
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

    plt.savefig(fn_out, bbox_inches="tight")
    plt.close()


def load_all_results(result_filepath):
    results = dict()
    with open(result_filepath, 'r') as f:
        for line in f.readlines()[1:]:
            split_lines = line.split(',')
            algo_name = split_lines[0].split('.')[0]
            if algo_name not in results:
                results[algo_name] = []
            results[algo_name].append([algo_name, float(split_lines[1]), 
                                  float(split_lines[2])])
    return results


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--result_csv", help="Path to CSV Results", required=True)
    parser.add_argument("--output", help="Path to the PNG output file",
                        default=f"{os.getcwd()}/out.png")
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

    print(f"writing output to {args.output}")

    results = load_all_results(args.result_csv)
    linestyles = create_linestyles(sorted(results.keys()))

    create_plot(results, args.raw, args.x_scale, args.y_scale, args.output, linestyles)


if __name__ == "__main__":
    main()
