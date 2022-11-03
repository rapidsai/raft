# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     h ttp://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

from pylibraft.neighbors import IvfPq

from pylibraft.testing.utils import TestDeviceBuffer


def generate_data(shape, dtype):
    if dtype in [np.float32]:
        x = np.random.random_sample(shape).astype(dtype)
    elif dtype == np.byte:
        x = np.random.randint(-127, 128, size=shape, dtype=np.byte)
    elif dtype == np.ubyte:
        x = np.random.randint(0, 255, size=shape, dtype=np.ubyte)
    return x


def calc_recall(ann_idx, true_nn_idx):
    assert ann_idx.shape == true_nn_idx.shape
    n = 0
    for i in range(ann_idx.shape[0]):
        n += np.intersect1d(ann_idx[i, :], true_nn_idx[i, :]).size
    recall = n / ann_idx.size
    return recall


def check_distances(dataset, queries, skl_metric, out_idx, out_dist):
    """
    Calculate the real distance between queries and dataset[out_idx], and compare it to out_dist.
    """
    dist = np.empty(out_dist.shape, out_dist.dtype)
    for i in range(queries.shape[0]):
        X = queries[np.newaxis, i, :]
        Y = dataset[out_idx[i, :], :]
        dist[i, :] = pairwise_distances(X, Y, skl_metric)

    if skl_metric == "euclidean":
        # Correct for differences in metric definition
        dist = np.power(dist, 2)

    dist_eps = abs(dist)
    dist_eps[dist < 1e-3] = 1e-3
    diff = abs(out_dist - dist) / dist_eps
    print(np.max(diff), np.min(diff), np.mean(diff), np.std(diff))

    # Quantization leads to errors in the distance calculation.
    # The aim of this test is not to test precision, but to catch obvious errors.
    assert np.mean(diff) < 0.1


def run_ivf_pq_build_search_test(
    n_rows,
    n_cols,
    n_queries,
    k,
    n_lists,
    metric,
    dtype,
    pq_bits,
    pq_dim,
    codebook_kind,
    force_random_rotation,
    add_data_on_build,
    n_probes,
    lut_dtype,
    internal_distance_dtype,
):
    dataset = generate_data((n_rows, n_cols), dtype)
    dataset_device = TestDeviceBuffer(dataset, order="C")

    nn = IvfPq()

    nn.build(
        dataset_device,
        n_lists,
        metric,
        kmeans_n_iters=20,
        kmeans_trainset_fraction=0.5,
        pq_bits=pq_bits,
        pq_dim=pq_dim,
        codebook_kind=codebook_kind,
        force_random_rotation=force_random_rotation,
        add_data_on_build=add_data_on_build,
    )

    if not add_data_on_build:
        assert False  # Extend interface not yet implemented

    assert nn._index is not None

    queries = generate_data((n_queries, n_cols), dtype)
    out_idx = np.zeros((n_queries, k), dtype=np.uint64)
    out_dist = np.zeros((n_queries, k), dtype=np.float32)

    queries_device = TestDeviceBuffer(queries, order="C")
    out_idx_device = TestDeviceBuffer(out_idx, order="C")
    out_dist_device = TestDeviceBuffer(out_dist, order="C")

    nn.search(
        queries_device,
        k,
        out_idx_device,
        out_dist_device,
        n_probes=n_probes,
        lut_dtype=lut_dtype,
        internal_distance_dtype=internal_distance_dtype,
    )

    out_idx = out_idx_device.copy_to_host()
    out_dist = out_dist_device.copy_to_host()

    # Calculate reference values with sklearn
    skl_metric = {"l2_expanded": "euclidean", "inner_product": "cosine"}[metric]
    # Note: raft l2 metric does not include the square root operation like sklearn's euclidean.
    # TODO(tfeher): document normalization diff between inner_product and cosine distance
    nn_skl = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=skl_metric)
    nn_skl.fit(dataset)
    skl_idx = nn_skl.kneighbors(queries, return_distance=False)

    recall = calc_recall(out_idx, skl_idx)
    assert recall > 0.8

    check_distances(dataset, queries, skl_metric, out_idx, out_dist)


# TODO(tfeher): Test over more parameters
@pytest.mark.parametrize("n_rows", [10000])
@pytest.mark.parametrize("n_cols", [10])
@pytest.mark.parametrize("n_queries", [100])
@pytest.mark.parametrize("n_lists", [100])
@pytest.mark.parametrize("metric", ["l2_expanded"])
@pytest.mark.parametrize("dtype", [np.float32, np.int8, np.uint8])
def test_ivf_pq_build(n_rows, n_cols, n_queries, n_lists, metric, dtype):

    run_ivf_pq_build_search_test(
        n_rows=n_rows,
        n_cols=n_cols,
        n_queries=n_queries,
        k=10,
        n_lists=n_lists,
        metric=metric,
        dtype=dtype,
        pq_bits=8,
        pq_dim=0,
        codebook_kind="per_subspace",
        force_random_rotation=False,
        add_data_on_build=True,
        n_probes=100,
        lut_dtype=IvfPq.CUDA_R_32F,
        internal_distance_dtype=IvfPq.CUDA_R_32F,
    )
