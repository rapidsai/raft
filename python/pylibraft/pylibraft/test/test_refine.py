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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from test_ivf_pq import calc_recall, check_distances, generate_data

from pylibraft.common import device_ndarray
from pylibraft.neighbors import refine


def run_refine(
    n_rows=500,
    n_cols=50,
    n_queries=100,
    metric="sqeuclidean",
    k0=40,
    k=10,
    inplace=False,
    dtype=np.float32,
    memory_type="device",
):

    dataset = generate_data((n_rows, n_cols), dtype)
    queries = generate_data((n_queries, n_cols), dtype)

    if metric == "inner_product":
        if dtype != np.float32:
            pytest.skip("Normalized input cannot be represented in int8")
            return
        dataset = normalize(dataset, norm="l2", axis=1)
        queries = normalize(queries, norm="l2", axis=1)

    dataset_device = device_ndarray(dataset)
    queries_device = device_ndarray(queries)

    # Calculate reference values with sklearn
    skl_metric = {"sqeuclidean": "euclidean", "inner_product": "cosine"}[
        metric
    ]
    nn_skl = NearestNeighbors(
        n_neighbors=k0, algorithm="brute", metric=skl_metric
    )
    nn_skl.fit(dataset)
    skl_dist, candidates = nn_skl.kneighbors(queries)
    candidates = candidates.astype(np.int64)
    candidates_device = device_ndarray(candidates)

    out_idx = np.zeros((n_queries, k), dtype=np.int64)
    out_dist = np.zeros((n_queries, k), dtype=np.float32)
    out_idx_device = device_ndarray(out_idx) if inplace else None
    out_dist_device = device_ndarray(out_dist) if inplace else None

    if memory_type == "device":
        if inplace:
            refine(
                dataset_device,
                queries_device,
                candidates_device,
                indices=out_idx_device,
                distances=out_dist_device,
                metric=metric,
            )
        else:
            out_dist_device, out_idx_device = refine(
                dataset_device,
                queries_device,
                candidates_device,
                k=k,
                metric=metric,
            )
        out_idx = out_idx_device.copy_to_host()
        out_dist = out_dist_device.copy_to_host()
    elif memory_type == "host":
        if inplace:
            refine(
                dataset,
                queries,
                candidates,
                indices=out_idx,
                distances=out_dist,
                metric=metric,
            )
        else:
            out_dist, out_idx = refine(
                dataset, queries, candidates, k=k, metric=metric
            )

    skl_idx = candidates[:, :k]

    recall = calc_recall(out_idx, skl_idx)
    if recall <= 0.999:
        # We did not find the same neighbor indices.
        # We could have found other neighbor with same distance.
        if metric == "sqeuclidean":
            skl_dist = np.power(skl_dist[:, :k], 2)
        elif metric == "inner_product":
            skl_dist = 1 - skl_dist[:, :k]
        else:
            raise ValueError("Invalid metric")
        mask = out_idx != skl_idx
        assert np.all(out_dist[mask] <= skl_dist[mask] + 1.0e-6)

    check_distances(dataset, queries, metric, out_idx, out_dist, 0.001)


@pytest.mark.parametrize("n_queries", [100, 1024, 37])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("metric", ["sqeuclidean", "inner_product"])
@pytest.mark.parametrize("dtype", [np.float32, np.int8, np.uint8])
@pytest.mark.parametrize("memory_type", ["device", "host"])
def test_refine_dtypes(n_queries, dtype, inplace, metric, memory_type):
    run_refine(
        n_rows=2000,
        n_queries=n_queries,
        n_cols=50,
        k0=40,
        k=10,
        dtype=dtype,
        inplace=inplace,
        metric=metric,
        memory_type=memory_type,
    )


@pytest.mark.parametrize(
    "params",
    [
        pytest.param(
            {
                "n_rows": 0,
                "n_cols": 10,
                "n_queries": 10,
                "k0": 10,
                "k": 1,
            },
            marks=pytest.mark.xfail(reason="empty dataset"),
        ),
        {"n_rows": 1, "n_cols": 10, "n_queries": 10, "k": 1, "k0": 1},
        {"n_rows": 10, "n_cols": 1, "n_queries": 10, "k": 10, "k0": 10},
        {"n_rows": 999, "n_cols": 42, "n_queries": 453, "k0": 137, "k": 53},
    ],
)
@pytest.mark.parametrize("memory_type", ["device", "host"])
def test_refine_row_col(params, memory_type):
    run_refine(
        n_rows=params["n_rows"],
        n_queries=params["n_queries"],
        n_cols=params["n_cols"],
        k0=params["k0"],
        k=params["k"],
        memory_type=memory_type,
    )


@pytest.mark.parametrize("memory_type", ["device", "host"])
def test_input_dtype(memory_type):
    with pytest.raises(Exception):
        run_refine(dtype=np.float64, memory_type=memory_type)


@pytest.mark.parametrize(
    "params",
    [
        {"idx_shape": None, "dist_shape": None, "k": None},
        {"idx_shape": [100, 9], "dist_shape": None, "k": 10},
        {"idx_shape": [101, 10], "dist_shape": None, "k": None},
        {"idx_shape": None, "dist_shape": [100, 11], "k": 10},
        {"idx_shape": None, "dist_shape": [99, 10], "k": None},
    ],
)
@pytest.mark.parametrize("memory_type", ["device", "host"])
def test_input_assertions(params, memory_type):
    n_cols = 5
    n_queries = 100
    k0 = 40
    dtype = np.float32
    dataset = generate_data((500, n_cols), dtype)
    dataset_device = device_ndarray(dataset)

    queries = generate_data((n_queries, n_cols), dtype)
    queries_device = device_ndarray(queries)

    candidates = np.random.randint(
        0, 500, size=(n_queries, k0), dtype=np.int64
    )
    candidates_device = device_ndarray(candidates)

    if params["idx_shape"] is not None:
        out_idx = np.zeros(params["idx_shape"], dtype=np.int64)
        out_idx_device = device_ndarray(out_idx)
    else:
        out_idx_device = None
    if params["dist_shape"] is not None:
        out_dist = np.zeros(params["dist_shape"], dtype=np.float32)
        out_dist_device = device_ndarray(out_dist)
    else:
        out_dist_device = None

    if memory_type == "device":
        with pytest.raises(Exception):
            distances, indices = refine(
                dataset_device,
                queries_device,
                candidates_device,
                k=params["k"],
                indices=out_idx_device,
                distances=out_dist_device,
            )
    else:
        with pytest.raises(Exception):
            distances, indices = refine(
                dataset,
                queries,
                candidates,
                k=params["k"],
                indices=out_idx,
                distances=out_dist,
            )
