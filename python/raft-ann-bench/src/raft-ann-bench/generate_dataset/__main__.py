#!/usr/bin/env python
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
#
import argparse
import math
import sys

import cuml
import cupy as cp

from ..generate_groundtruth.utils import memmap_bin_file


def generate_dataset(filename, n_samples, n_features, dtype, rng):
    fp = memmap_bin_file(
        filename, dtype, shape=(n_samples, n_features), mode="w+"
    )
    dtype = fp.dtype
    itemsize = fp.dtype.itemsize
    total_size = n_samples * n_features * itemsize / (1 << 30)

    print(
        "Generating dataset {0} shape=({1},{2}), dtype={3}, size={4:6.1f} "
        "GiB".format(filename, n_samples, n_features, dtype, total_size)
    )

    n = 1000000
    i = 0
    while i < n_samples:
        n_batch = n if i + n <= n_samples else n_samples - i
        if rng == "blobs":
            tmp, y = cuml.datasets.make_blobs(
                n_batch,
                n_features,
                centers=int(math.sqrt(n_samples)),
                cluster_std=3,
                shuffle=True,
                random_state=1234,
                order="C",
            )
            tmp = tmp.astype(dtype)
        else:
            tmp = cp.random.uniform(size=(n_batch, n_features)).astype(dtype)
        fp[i : i + n_batch, :] = cp.asnumpy(tmp)
        i += n_batch
        print(
            "Step {0}/{1}: {2:6.1f} GiB written".format(
                i // n, n_samples // n, i * n_features * itemsize / (1 << 30)
            )
        )

    fp.flush()
    del fp


def main():
    parser = argparse.ArgumentParser(
        prog="generate_dataset",
        description="Generate random dataset. "
        "The output file is in big-ann-benchmark's binary format.",
        epilog="""Example usage
    python -m raft-ann-bench.generate_dataset --rows 1000000 --cols 128\
 --dtype float32 dataset/base.fbin

 # After the dataset is generated, you can create query and ground truth files

 python -m raft-ann-bench.generate_groundtruth dataset/base.fbin\
 --output=/dataset --queries=random --n_queries=10000
    """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("filename", type=str, help="output dataset file name")
    parser.add_argument(
        "--rng",
        type=str,
        default="blobs",
        help="Random generator to use, one of 'uniform' or 'blobs' "
        "(default).",
    )

    parser.add_argument(
        "-N",
        "--rows",
        default=1000000,
        type=int,
        help="Number of rows to generate (default 1M)",
    )
    parser.add_argument(
        "-D",
        "--cols",
        default=128,
        type=int,
        help="number of features (dataset columns, default 128)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        help="Dataset dtype. When not specified, then derived from extension."
        " Supported types: 'float32', 'float16', 'uint8', 'int8'",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    generate_dataset(
        args.filename, args.rows, args.cols, args.dtype, rng=args.rng
    )


if __name__ == "__main__":
    main()
