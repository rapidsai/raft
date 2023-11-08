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
import os
import sys

import cupy as cp
import numpy as np
import rmm
from pylibraft.common import DeviceResources
from pylibraft.neighbors.brute_force import knn
from rmm.allocators.cupy import rmm_cupy_allocator

from .utils import memmap_bin_file, suffix_from_dtype, write_bin


def generate_random_queries(n_queries, n_features, dtype=np.float32):
    print("Generating random queries")
    if np.issubdtype(dtype, np.integer):
        queries = cp.random.randint(
            0, 255, size=(n_queries, n_features), dtype=dtype
        )
    else:
        queries = cp.random.uniform(size=(n_queries, n_features)).astype(dtype)
    return queries


def choose_random_queries(dataset, n_queries):
    print("Choosing random vector from dataset as query vectors")
    query_idx = np.random.choice(
        dataset.shape[0], size=(n_queries,), replace=False
    )
    return dataset[query_idx, :]


def calc_truth(dataset, queries, k, metric="sqeuclidean"):
    handle = DeviceResources()
    n_samples = dataset.shape[0]
    n = 500000  # batch size for processing neighbors
    i = 0
    indices = None
    distances = None
    queries = cp.asarray(queries, dtype=cp.float32)

    while i < n_samples:
        print("Step {0}/{1}:".format(i // n, n_samples // n))
        n_batch = n if i + n <= n_samples else n_samples - i

        X = cp.asarray(dataset[i : i + n_batch, :], cp.float32)

        D, Ind = knn(
            X,
            queries,
            k,
            metric=metric,
            handle=handle,
            global_id_offset=i,  # shift neighbor index by offset i
        )
        handle.sync()

        D, Ind = cp.asarray(D), cp.asarray(Ind)
        if distances is None:
            distances = D
            indices = Ind
        else:
            distances = cp.concatenate([distances, D], axis=1)
            indices = cp.concatenate([indices, Ind], axis=1)
            idx = cp.argsort(distances, axis=1)[:, :k]
            distances = cp.take_along_axis(distances, idx, axis=1)
            indices = cp.take_along_axis(indices, idx, axis=1)

        i += n_batch

    return distances, indices


def main():
    pool = rmm.mr.PoolMemoryResource(
        rmm.mr.CudaMemoryResource(), initial_pool_size=2**30
    )
    rmm.mr.set_current_device_resource(pool)
    cp.cuda.set_allocator(rmm_cupy_allocator)

    parser = argparse.ArgumentParser(
        prog="generate_groundtruth",
        description="Generate true neighbors using exact NN search. "
        "The input and output files are in big-ann-benchmark's binary format.",
        epilog="""Example usage
    # With existing query file
    python -m raft-ann-bench.generate_groundtruth --dataset /dataset/base.\
fbin --output=groundtruth_dir --queries=/dataset/query.public.10K.fbin

    # With randomly generated queries
    python -m raft-ann-bench.generate_groundtruth --dataset /dataset/base.\
fbin --output=groundtruth_dir --queries=random --n_queries=10000

    # Using only a subset of the dataset. Define queries by randomly
    # selecting vectors from the (subset of the) dataset.
    python -m raft-ann-bench.generate_groundtruth --dataset /dataset/base.\
fbin --nrows=2000000 --cols=128 --output=groundtruth_dir \
--queries=random-choice --n_queries=10000
    """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("dataset", type=str, help="input dataset file name")
    parser.add_argument(
        "--queries",
        type=str,
        default="random",
        help="Queries file name, or one of 'random-choice' or 'random' "
        "(default). 'random-choice': select n_queries vectors from the input "
        "dataset. 'random': generate n_queries as uniform random numbers.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="output directory name (default current dir)",
    )

    parser.add_argument(
        "--n_queries",
        type=int,
        default=10000,
        help="Number of quries to generate (if no query file is given). "
        "Default: 10000.",
    )

    parser.add_argument(
        "-N",
        "--rows",
        default=None,
        type=int,
        help="use only first N rows from dataset, by default the whole "
        "dataset is used",
    )
    parser.add_argument(
        "-D",
        "--cols",
        default=None,
        type=int,
        help="number of features (dataset columns). "
        "Default: read from dataset file.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        help="Dataset dtype. When not specified, then derived from extension."
        " Supported types: 'float32', 'float16', 'uint8', 'int8'",
    )

    parser.add_argument(
        "-k",
        type=int,
        default=100,
        help="Number of neighbors (per query) to calculate",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="sqeuclidean",
        help="Metric to use while calculating distances. Valid metrics are "
        "those that are accepted by pylibraft.neighbors.brute_force.knn. Most"
        " commonly used with RAFT ANN are 'sqeuclidean' and 'inner_product'",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    if args.rows is not None:
        print("Reading subset of the data, nrows=", args.rows)
    else:
        print("Reading whole dataset")

    # Load input data
    dataset = memmap_bin_file(
        args.dataset, args.dtype, shape=(args.rows, args.cols)
    )
    n_features = dataset.shape[1]
    dtype = dataset.dtype

    print(
        "Dataset size {:6.1f} GB, shape {}, dtype {}".format(
            dataset.size * dataset.dtype.itemsize / 1e9,
            dataset.shape,
            np.dtype(dtype),
        )
    )

    if len(args.output) > 0:
        os.makedirs(args.output, exist_ok=True)

    if args.queries == "random" or args.queries == "random-choice":
        if args.n_queries is None:
            raise RuntimeError(
                "n_queries must be given to generate random queries"
            )
        if args.queries == "random":
            queries = generate_random_queries(
                args.n_queries, n_features, dtype
            )
        elif args.queries == "random-choice":
            queries = choose_random_queries(dataset, args.n_queries)

        queries_filename = os.path.join(
            args.output, "queries" + suffix_from_dtype(dtype)
        )
        print("Writing queries file", queries_filename)
        write_bin(queries_filename, queries)
    else:
        print("Reading queries from file", args.queries)
        queries = memmap_bin_file(args.queries, dtype)

    print("Calculating true nearest neighbors")
    distances, indices = calc_truth(dataset, queries, args.k, args.metric)

    write_bin(
        os.path.join(args.output, "groundtruth.neighbors.ibin"),
        indices.astype(np.uint32),
    )
    write_bin(
        os.path.join(args.output, "groundtruth.distances.fbin"),
        distances.astype(np.float32),
    )


if __name__ == "__main__":
    main()
