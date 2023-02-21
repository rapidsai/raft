# Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
from pylibraft.neighbors import ivf_pq


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


def run_ivf_pq_build_search_test(
    n_rows,
    n_cols,
    n_queries,
    k,
    n_lists,
    metric,
    dtype,
    pq_bits=8,
    pq_dim=0,
    codebook_kind="subspace",
    add_data_on_build="True",
    n_probes=100,
    lut_dtype=np.float32,
    internal_distance_dtype=np.float32,
    force_random_rotation=False,
    kmeans_trainset_fraction=1,
    kmeans_n_iters=20,
    compare=True,
    inplace=True,
    array_type="device",
):
    dataset = generate_data((n_rows, n_cols), dtype)
    if metric == "inner_product":
        dataset = normalize(dataset, norm="l2", axis=1)
    dataset_device = device_ndarray(dataset)

    build_params = ivf_pq.IndexParams(
        n_lists=n_lists,
        metric=metric,
        kmeans_n_iters=kmeans_n_iters,
        kmeans_trainset_fraction=kmeans_trainset_fraction,
        pq_bits=pq_bits,
        pq_dim=pq_dim,
        codebook_kind=codebook_kind,
        force_random_rotation=force_random_rotation,
        add_data_on_build=add_data_on_build,
    )

    if array_type == "device":
        index = ivf_pq.build(build_params, dataset_device)
    else:
        index = ivf_pq.build(build_params, dataset)

    assert index.trained
    if pq_dim != 0:
        assert index.pq_dim == build_params.pq_dim
    assert index.pq_bits == build_params.pq_bits
    assert index.metric == build_params.metric
    assert index.n_lists == build_params.n_lists

    if not add_data_on_build:
        dataset_1 = dataset[: n_rows // 2, :]
        dataset_2 = dataset[n_rows // 2 :, :]
        indices_1 = np.arange(n_rows // 2, dtype=np.uint64)
        indices_2 = np.arange(n_rows // 2, n_rows, dtype=np.uint64)
        if array_type == "device":
            dataset_1_device = device_ndarray(dataset_1)
            dataset_2_device = device_ndarray(dataset_2)
            indices_1_device = device_ndarray(indices_1)
            indices_2_device = device_ndarray(indices_2)
            index = ivf_pq.extend(index, dataset_1_device, indices_1_device)
            index = ivf_pq.extend(index, dataset_2_device, indices_2_device)
        else:
            index = ivf_pq.extend(index, dataset_1, indices_1)
            index = ivf_pq.extend(index, dataset_2, indices_2)

    assert index.size >= n_rows

    queries = generate_data((n_queries, n_cols), dtype)
    out_idx = np.zeros((n_queries, k), dtype=np.uint64)
    out_dist = np.zeros((n_queries, k), dtype=np.float32)

    queries_device = device_ndarray(queries)
    out_idx_device = device_ndarray(out_idx) if inplace else None
    out_dist_device = device_ndarray(out_dist) if inplace else None

    search_params = ivf_pq.SearchParams(
        n_probes=n_probes,
        lut_dtype=lut_dtype,
        internal_distance_dtype=internal_distance_dtype,
    )

    ret_output = ivf_pq.search(
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
@pytest.mark.parametrize("array_type", ["host", "device"])
def test_ivf_pq_dtypes(
    n_rows, n_cols, n_queries, n_lists, dtype, inplace, array_type
):
    # Note that inner_product tests use normalized input which we cannot
    # represent in int8, therefore we test only sqeuclidean metric here.
    run_ivf_pq_build_search_test(
        n_rows=n_rows,
        n_cols=n_cols,
        n_queries=n_queries,
        k=10,
        n_lists=n_lists,
        metric="sqeuclidean",
        dtype=dtype,
        inplace=inplace,
        array_type=array_type,
    )


@pytest.mark.parametrize(
    "params",
    [
        pytest.param(
            {
                "n_rows": 0,
                "n_cols": 10,
                "n_queries": 10,
                "k": 1,
                "n_lists": 10,
            },
            marks=pytest.mark.xfail(reason="empty dataset"),
        ),
        {"n_rows": 1, "n_cols": 10, "n_queries": 10, "k": 1, "n_lists": 1},
        {"n_rows": 10, "n_cols": 1, "n_queries": 10, "k": 10, "n_lists": 10},
        # {"n_rows": 999, "n_cols": 42, "n_queries": 453, "k": 137,
        #  "n_lists": 53},
    ],
)
def test_ivf_pq_n(params):
    # We do not test recall, just confirm that we can handle edge cases for
    # certain parameters
    run_ivf_pq_build_search_test(
        n_rows=params["n_rows"],
        n_cols=params["n_cols"],
        n_queries=params["n_queries"],
        k=params["k"],
        n_lists=params["n_lists"],
        metric="sqeuclidean",
        dtype=np.float32,
        compare=False,
    )


@pytest.mark.parametrize(
    "metric", ["sqeuclidean", "inner_product", "euclidean"]
)
@pytest.mark.parametrize("dtype", [np.float32])
@pytest.mark.parametrize("codebook_kind", ["subspace", "cluster"])
@pytest.mark.parametrize("rotation", [True, False])
def test_ivf_pq_build_params(metric, dtype, codebook_kind, rotation):
    run_ivf_pq_build_search_test(
        n_rows=10000,
        n_cols=10,
        n_queries=1000,
        k=10,
        n_lists=100,
        metric=metric,
        dtype=dtype,
        pq_bits=8,
        pq_dim=0,
        codebook_kind=codebook_kind,
        add_data_on_build=True,
        n_probes=100,
        force_random_rotation=rotation,
    )


@pytest.mark.parametrize(
    "params",
    [
        {"pq_dims": 10, "pq_bits": 8, "n_lists": 100},
        {"pq_dims": 16, "pq_bits": 7, "n_lists": 100},
        {"pq_dims": 0, "pq_bits": 8, "n_lists": 90},
        {
            "pq_dims": 0,
            "pq_bits": 8,
            "n_lists": 100,
            "trainset_fraction": 0.9,
            "n_iters": 30,
        },
    ],
)
def test_ivf_pq_params(params):
    run_ivf_pq_build_search_test(
        n_rows=10000,
        n_cols=16,
        n_queries=1000,
        k=10,
        n_lists=params["n_lists"],
        metric="sqeuclidean",
        dtype=np.float32,
        pq_bits=params["pq_bits"],
        pq_dim=params["pq_dims"],
        kmeans_trainset_fraction=params.get("trainset_fraction", 1.0),
        kmeans_n_iters=params.get("n_iters", 20),
    )


@pytest.mark.parametrize(
    "params",
    [
        {
            "k": 10,
            "n_probes": 100,
            "lut": np.float16,
            "idd": np.float32,
        },
        {
            "k": 10,
            "n_probes": 99,
            "lut": np.uint8,
            "idd": np.float32,
        },
        {
            "k": 10,
            "n_probes": 100,
            "lut": np.float16,
            "idd": np.float16,
        },
        {
            "k": 129,
            "n_probes": 100,
            "lut": np.float32,
            "idd": np.float32,
        },
    ],
)
def test_ivf_pq_search_params(params):
    run_ivf_pq_build_search_test(
        n_rows=10000,
        n_cols=16,
        n_queries=1000,
        k=params["k"],
        n_lists=100,
        n_probes=params["n_probes"],
        metric="sqeuclidean",
        dtype=np.float32,
        lut_dtype=params["lut"],
        internal_distance_dtype=params["idd"],
    )


@pytest.mark.parametrize("dtype", [np.float32, np.int8, np.uint8])
@pytest.mark.parametrize("array_type", ["host", "device"])
def test_extend(dtype, array_type):
    run_ivf_pq_build_search_test(
        n_rows=10000,
        n_cols=10,
        n_queries=100,
        k=10,
        n_lists=100,
        metric="sqeuclidean",
        dtype=dtype,
        add_data_on_build=False,
        array_type=array_type,
    )


def test_build_assertions():
    with pytest.raises(TypeError):
        run_ivf_pq_build_search_test(
            n_rows=1000,
            n_cols=10,
            n_queries=100,
            k=10,
            n_lists=100,
            metric="sqeuclidean",
            dtype=np.float64,
        )

    n_rows = 1000
    n_cols = 100
    n_queries = 212
    k = 10
    dataset = generate_data((n_rows, n_cols), np.float32)
    dataset_device = device_ndarray(dataset)

    index_params = ivf_pq.IndexParams(
        n_lists=50,
        metric="sqeuclidean",
        kmeans_n_iters=20,
        kmeans_trainset_fraction=1,
        add_data_on_build=False,
    )

    index = ivf_pq.Index()

    queries = generate_data((n_queries, n_cols), np.float32)
    out_idx = np.zeros((n_queries, k), dtype=np.uint64)
    out_dist = np.zeros((n_queries, k), dtype=np.float32)

    queries_device = device_ndarray(queries)
    out_idx_device = device_ndarray(out_idx)
    out_dist_device = device_ndarray(out_dist)

    search_params = ivf_pq.SearchParams(n_probes=50)

    with pytest.raises(ValueError):
        # Index must be built before search
        ivf_pq.search(
            search_params,
            index,
            queries_device,
            k,
            out_idx_device,
            out_dist_device,
        )

    index = ivf_pq.build(index_params, dataset_device)
    assert index.trained

    indices = np.arange(n_rows + 1, dtype=np.uint64)
    indices_device = device_ndarray(indices)

    with pytest.raises(ValueError):
        # Dataset dimension mismatch
        ivf_pq.extend(index, queries_device, indices_device)

    with pytest.raises(ValueError):
        # indices dimension mismatch
        ivf_pq.extend(index, dataset_device, indices_device)


@pytest.mark.parametrize(
    "params",
    [
        {"q_dt": np.float64},
        {"q_order": "F"},
        {"q_cols": 101},
        {"idx_dt": np.uint32},
        {"idx_order": "F"},
        {"idx_rows": 42},
        {"idx_cols": 137},
        {"dist_dt": np.float64},
        {"dist_order": "F"},
        {"dist_rows": 42},
        {"dist_cols": 137},
    ],
)
def test_search_inputs(params):
    """Test with invalid input dtype, order, or dimension."""
    n_rows = 1000
    n_cols = 100
    n_queries = 256
    k = 10
    dtype = np.float32

    q_dt = params.get("q_dt", np.float32)
    q_order = params.get("q_order", "C")
    queries = generate_data(
        (n_queries, params.get("q_cols", n_cols)), q_dt
    ).astype(q_dt, order=q_order)
    queries_device = device_ndarray(queries)

    idx_dt = params.get("idx_dt", np.uint64)
    idx_order = params.get("idx_order", "C")
    out_idx = np.zeros(
        (params.get("idx_rows", n_queries), params.get("idx_cols", k)),
        dtype=idx_dt,
        order=idx_order,
    )
    out_idx_device = device_ndarray(out_idx)

    dist_dt = params.get("dist_dt", np.float32)
    dist_order = params.get("dist_order", "C")
    out_dist = np.zeros(
        (params.get("dist_rows", n_queries), params.get("dist_cols", k)),
        dtype=dist_dt,
        order=dist_order,
    )
    out_dist_device = device_ndarray(out_dist)

    index_params = ivf_pq.IndexParams(
        n_lists=50, metric="sqeuclidean", add_data_on_build=True
    )

    dataset = generate_data((n_rows, n_cols), dtype)
    dataset_device = device_ndarray(dataset)
    index = ivf_pq.build(index_params, dataset_device)
    assert index.trained

    with pytest.raises(Exception):
        search_params = ivf_pq.SearchParams(n_probes=50)
        ivf_pq.search(
            search_params,
            index,
            queries_device,
            k,
            out_idx_device,
            out_dist_device,
        )


def test_save_load():
    n_rows = 10000
    n_cols = 50
    n_queries = 1000
    dtype = np.float32

    dataset = generate_data((n_rows, n_cols), dtype)
    dataset_device = device_ndarray(dataset)

    build_params = ivf_pq.IndexParams(n_lists=100, metric="sqeuclidean")
    index = ivf_pq.build(build_params, dataset_device)

    assert index.trained
    filename = "my_index.bin"
    ivf_pq.save(filename, index)
    loaded_index = ivf_pq.load(filename)

    assert index.pq_dim == loaded_index.pq_dim
    assert index.pq_bits == loaded_index.pq_bits
    assert index.metric == loaded_index.metric
    assert index.n_lists == loaded_index.n_lists
    assert index.size == loaded_index.size

    queries = generate_data((n_queries, n_cols), dtype)

    queries_device = device_ndarray(queries)
    search_params = ivf_pq.SearchParams(n_probes=100)
    k = 10

    distance_dev, neighbors_dev = ivf_pq.search(
        search_params, index, queries_device, k
    )

    neighbors = neighbors_dev.copy_to_host()
    dist = distance_dev.copy_to_host()
    del index

    distance_dev, neighbors_dev = ivf_pq.search(
        search_params, loaded_index, queries_device, k
    )

    neighbors2 = neighbors_dev.copy_to_host()
    dist2 = distance_dev.copy_to_host()

    assert np.all(neighbors == neighbors2)
    assert np.allclose(dist, dist2, rtol=1e-6)
