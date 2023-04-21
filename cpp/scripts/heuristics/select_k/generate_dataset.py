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
#

import timeit
import cupy as cp
import numpy as np
from pylibraft.matrix import SelectMethod, select_k


# limit size of the dataset being selected to 32GB for now
MAX_MEMORY = 16 * 1024 * 1024 * 1024


def get_times(func, min_number=5, max_number=128, max_time=0.5):
    timer = timeit.Timer(func)
    times = []

    total = 0

    while True:
        t = timer.timeit(number=1)
        times.append(t)
        total += t
        if (total >= max_time and len(times) >= min_number) or len(
            times
        ) >= max_number:
            break

    return min(times)


def generate_dataset(
    output_filename="select_k_times.csv",
    append=False,
    algos=None,
    include_grid=True,
    include_random=True,
):
    features = []

    # generate features in a grid across our input ranges
    # (this is required for generating plots right now, since
    # we fix 2 of the inputs to show changes in the 3rd)
    if include_grid:
        k_vals = [2**i for i in range(14)]
        n_rows = [2**i for i in range(16)]
        n_cols = [2**i for i in range(10, 28)]

        # also insert values just after the boundaries for different algorithms
        # (makes heuristic learning learn the hard limit here)
        k_vals.extend([257, 2049])

        for k in k_vals:
            for row in n_rows:
                for col in n_cols:
                    features.append((k, row, col))

    # also generate 5k random rows too (in addition to the regular grid)
    # to help with learning heuristics
    if include_random:
        for _ in range(5000):
            features.append(
                (
                    int(2 ** np.random.uniform(14)),
                    int(2 ** np.random.uniform(16)),
                    int(2 ** np.random.uniform(10, 28)),
                )
            )

    # filter out invalid values
    features = [
        (k, row, col)
        for k, row, col in features
        if k <= col and row * col * 4 < MAX_MEMORY
    ]

    algos = algos or [
        SelectMethod.RADIX,
        SelectMethod.WARPSORT,
        SelectMethod.BLOCK,
    ]

    if append:
        output = open(output_filename, "a")
    else:
        output = open(output_filename, "w")
        output.write("algo,k,row,col,time\n")

    for k, row, col in features:
        dataset = cp.random.randn(row, col, dtype="float32")
        for algo in algos:
            if (algo == SelectMethod.WARPSORT and k > 256) or (
                algo == SelectMethod.BLOCK and k > 2048
            ):
                time = np.inf
            else:
                time = get_times(lambda: select_k(dataset, k=k, algo=algo))

            line = f"{str(algo).split('.')[1]},{k},{row},{col},{time:0.8f}"
            print(line)
            output.write(line + "\n")
        dataset = None
    output.close()


if __name__ == "__main__":
    generate_dataset()
