# Copyright (c) 2023, NVIDIA CORPORATION.
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

import numpy as np
import pytest
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from pylibraft.common import device_ndarray
from pylibraft.neighbors import cagra


def generate_data(shape, dtype):
    if dtype == np.byte:
        x = np.random.randint(-127, 128, size=shape, dtype=np.byte)
    elif dtype == np.ubyte:
        x = np.random.randint(0, 255, size=shape, dtype=np.ubyte)
    else:
        x = np.random.random_sample(shape).astype(dtype)

    return x


def calc_recall(ann_idx, true_nn_idx):
    assert ann_idx.shape == true_nn_idx.shape
    n = 0
    for i in range(ann_idx.shape[0]):
        n += np.intersect1d(ann_idx[i, :], true_nn_idx[i, :]).size
    recall = n / ann_idx.size
    return recall


def check_distances(dataset, queries, metric, out_idx, out_dist, eps=None):
    """
    Calculate the real distance between queries and dataset[out_idx],
    and compare it to out_dist.
    """
    if eps is None:
        # Quantization leads to errors in the distance calculation.
        # The aim of this test is not to test precision, but to catch obvious
        # errors.
        eps = 0.1

    dist = np.empty(out_dist.shape, out_dist.dtype)
    for i in range(queries.shape[0]):
        X = queries[np.newaxis, i, :]
        Y = dataset[out_idx[i, :], :]
        if metric == "sqeuclidean":
            dist[i, :] = pairwise_distances(X, Y, "sqeuclidean")
        elif metric == "euclidean":
            dist[i, :] = pairwise_distances(X, Y, "euclidean")
        elif metric == "inner_product":
            dist[i, :] = np.matmul(X, Y.T)
        else:
            raise ValueError("Invalid metric")

    dist_eps = abs(dist)
    dist_eps[dist < 1e-3] = 1e-3
    diff = abs(out_dist - dist) / dist_eps

    assert np.mean(diff) < eps


def run_cagra_build_search_test(
    n_rows,
    n_cols,
    n_queries,
    k,
    dtype,
    metric,
    array_type="device",
    compare=True,
    inplace=True,
    add_data_on_build=True,
):
    dataset = generate_data((n_rows, n_cols), dtype)
    if metric == "inner_product":
        dataset = normalize(dataset, norm="l2", axis=1)
    dataset_device = device_ndarray(dataset)

    build_params = cagra.IndexParams(

    )

    if array_type == "device":
        index = cagra.build(build_params, dataset_device)
    else:
        index = cagra.build(build_params, dataset)

    assert index.trained

    if not add_data_on_build:
        dataset_1 = dataset[: n_rows // 2, :]
        dataset_2 = dataset[n_rows // 2 :, :]
        indices_1 = np.arange(n_rows // 2, dtype=np.uint32)
        indices_2 = np.arange(n_rows // 2, n_rows, dtype=np.uint32)
        if array_type == "device":
            dataset_1_device = device_ndarray(dataset_1)
            dataset_2_device = device_ndarray(dataset_2)
            indices_1_device = device_ndarray(indices_1)
            indices_2_device = device_ndarray(indices_2)
            index = cagra.extend(index, dataset_1_device, indices_1_device)
            index = cagra.extend(index, dataset_2_device, indices_2_device)
        else:
            index = cagra.extend(index, dataset_1, indices_1)
            index = cagra.extend(index, dataset_2, indices_2)

    queries = generate_data((n_queries, n_cols), dtype)
    out_idx = np.zeros((n_queries, k), dtype=np.uint32)
    out_dist = np.zeros((n_queries, k), dtype=np.float32)

    queries_device = device_ndarray(queries)
    out_idx_device = device_ndarray(out_idx) if inplace else None
    out_dist_device = device_ndarray(out_dist) if inplace else None

    search_params = cagra.SearchParams()

    ret_output = cagra.search(
        search_params,
        index,
        queries_device,
        k,
        neighbors=out_idx_device,
        distances=out_dist_device,
    )

    if not inplace:
        out_dist_device, out_idx_device = ret_output

    if not compare:
        return

    out_idx = out_idx_device.copy_to_host()
    out_dist = out_dist_device.copy_to_host()

    # Calculate reference values with sklearn
    skl_metric = {
        "sqeuclidean": "sqeuclidean",
        "inner_product": "cosine",
        "euclidean": "euclidean",
    }[metric]
    nn_skl = NearestNeighbors(
        n_neighbors=k, algorithm="brute", metric=skl_metric
    )
    nn_skl.fit(dataset)
    skl_idx = nn_skl.kneighbors(queries, return_distance=False)

    recall = calc_recall(out_idx, skl_idx)
    assert recall > 0.7

    check_distances(dataset, queries, metric, out_idx, out_dist)


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("n_rows", [10000])
@pytest.mark.parametrize("n_cols", [10])
@pytest.mark.parametrize("n_queries", [100])
@pytest.mark.parametrize("n_lists", [100])
@pytest.mark.parametrize("dtype", [np.float32, np.int8, np.uint8])
@pytest.mark.parametrize("array_type", ["device"])
def test_cagra_build_search(
    n_rows, n_cols, n_queries, n_lists, dtype, inplace, array_type
):
    # Note that inner_product tests use normalized input which we cannot
    # represent in int8, therefore we test only sqeuclidean metric here.
    run_cagra_build_search_test(
        n_rows=n_rows,
        n_cols=n_cols,
        n_queries=n_queries,
        k=10,
        dtype=dtype,
        inplace=inplace,
        array_type=array_type,
        metric="sqeuclidean",
    )


@pytest.mark.parametrize("dtype", [np.float32, np.int8, np.ubyte])
def test_save_load(dtype):
    n_rows = 10000
    n_cols = 50
    n_queries = 1000

    dataset = generate_data((n_rows, n_cols), dtype)
    dataset_device = device_ndarray(dataset)

    build_params = cagra.IndexParams()
    index = cagra.build(build_params, dataset_device)

    assert index.trained
    filename = "my_index.bin"
    cagra.save(filename, index)
    loaded_index = cagra.load(filename)

    queries = generate_data((n_queries, n_cols), dtype)

    queries_device = device_ndarray(queries)
    search_params = cagra.SearchParams()
    k = 10

    distance_dev, neighbors_dev = cagra.search(
        search_params, index, queries_device, k
    )

    neighbors = neighbors_dev.copy_to_host()
    dist = distance_dev.copy_to_host()
    del index

    distance_dev, neighbors_dev = cagra.search(
        search_params, loaded_index, queries_device, k
    )

    neighbors2 = neighbors_dev.copy_to_host()
    dist2 = distance_dev.copy_to_host()

    assert np.all(neighbors == neighbors2)
    assert np.allclose(dist, dist2, rtol=1e-6)
