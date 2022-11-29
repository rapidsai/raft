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
    metric="l2_expanded",
    k0=40,
    k=10,
    inplace=False,
    dtype=np.float32,
):

    dataset = generate_data((n_rows, n_cols), dtype)
    if metric == "inner_product":
        if dtype != np.float32:
            pytest.skip("Normalized input cannot be represented in int8")
            return
        dataset = normalize(dataset, norm="l2", axis=1)
    dataset_device = device_ndarray(dataset)

    queries = generate_data((n_queries, n_cols), dtype)
    queries_device = device_ndarray(queries)

    # Calculate reference values with sklearn
    skl_metric = {"l2_expanded": "euclidean", "inner_product": "cosine"}[
        metric
    ]
    nn_skl = NearestNeighbors(
        n_neighbors=k0, algorithm="brute", metric=skl_metric
    )
    nn_skl.fit(dataset)
    candidates = nn_skl.kneighbors(queries, return_distance=False).astype(
        np.uint64
    )
    candidates_device = device_ndarray(candidates)

    out_idx = np.zeros((n_queries, k), dtype=np.uint64)
    out_dist = np.zeros((n_queries, k), dtype=np.float32)
    out_idx_device = device_ndarray(out_idx) if inplace else None
    out_dist_device = device_ndarray(out_dist) if inplace else None

    if inplace:
        refine(
            dataset_device,
            queries_device,
            candidates_device,
            indices=out_idx_device,
            distances=out_dist_device,
        )
    else:
        out_dist_device, out_idx_device = refine(
            dataset_device, queries_device, candidates_device, k=k
        )

    out_idx = out_idx_device.copy_to_host()
    out_dist = out_dist_device.copy_to_host()

    skl_idx = candidates[:, :k]
    recall = calc_recall(out_idx, skl_idx)
    print(recall)
    assert recall > 0.999

    check_distances(dataset, queries, metric, out_idx, out_dist)


@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize("metric", ["l2_expanded", "inner_product"])
@pytest.mark.parametrize("dtype", [np.float32, np.int8, np.uint8])
def test_refine_dtypes(dtype, inplace, metric):
    run_refine(
        n_rows=5000,
        n_queries=10,
        n_cols=5,
        k0=40,
        k=10,
        dtype=dtype,
        inplace=inplace,
        metric=metric,
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
def test_refine_row_col(params):
    run_refine(
        n_rows=params["n_rows"],
        n_queries=params["n_queries"],
        n_cols=params["n_cols"],
        k0=params["k0"],
        k=params["k"],
    )


def test_input_dtype():
    with pytest.raises(Exception):
        run_refine(dtype=np.float64)


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
def test_input_assertions(params):
    n_cols = 5
    n_queries = 100
    k0 = 40
    dtype = np.float32
    dataset = generate_data((500, n_cols), dtype)
    dataset_device = device_ndarray(dataset)

    queries = generate_data((n_queries, n_cols), dtype)
    queries_device = device_ndarray(queries)

    candidates = np.random.randint(
        0, 500, size=(n_queries, k0), dtype=np.uint64
    )
    candidates_device = device_ndarray(candidates)

    if params["idx_shape"] is not None:
        out_idx = np.zeros(params["idx_shape"], dtype=np.uint64)
        out_idx_device = device_ndarray(out_idx)
    else:
        out_idx_device = None
    if params["dist_shape"] is not None:
        out_dist = np.zeros(params["dist_shape"], dtype=np.float32)
        out_dist_device = device_ndarray(out_dist)
    else:
        out_dist_device = None

    with pytest.raises(Exception):
        distances, indices = refine(
            dataset_device,
            queries_device,
            candidates_device,
            k=params["k"],
            indices=out_idx_device,
            distances=out_dist_device,
        )
