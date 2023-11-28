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
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from pylibraft.neighbors import cagra, cagra_hnswlib
from pylibraft.test.ann_utils import calc_recall, generate_data


def run_cagra_hnswlib_build_search_test(
    n_rows=10000,
    n_cols=10,
    n_queries=100,
    k=10,
    dtype=np.float32,
    metric="sqeuclidean",
    intermediate_graph_degree=128,
    graph_degree=64,
    search_params={},
):
    dataset = generate_data((n_rows, n_cols), dtype)
    if metric == "inner_product":
        dataset = normalize(dataset, norm="l2", axis=1)

    build_params = cagra.IndexParams(
        metric=metric,
        intermediate_graph_degree=intermediate_graph_degree,
        graph_degree=graph_degree,
    )

    index = cagra.build(build_params, dataset)

    assert index.trained

    filename = "my_index.bin"
    cagra_hnswlib.save(filename, index)

    index_hnswlib = cagra_hnswlib.load(
        filename, n_cols, dataset.dtype, metric=metric
    )

    queries = generate_data((n_queries, n_cols), dtype)
    out_idx = np.zeros((n_queries, k), dtype=np.uint32)

    search_params = cagra_hnswlib.SearchParams(**search_params)

    out_dist, out_idx = cagra_hnswlib.search(
        search_params, index_hnswlib, queries, k
    )

    # Calculate reference values with sklearn
    nn_skl = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    nn_skl.fit(dataset)
    skl_idx = nn_skl.kneighbors(queries, return_distance=False)

    recall = calc_recall(out_idx, skl_idx)
    print(recall)
    assert recall > 0.95


@pytest.mark.parametrize("dtype", [np.float32, np.int8, np.uint8])
@pytest.mark.parametrize("k", [10, 20])
@pytest.mark.parametrize("ef", [30, 40])
@pytest.mark.parametrize("num_threads", [2, 4])
def test_cagra_hnswlib(dtype, k, ef, num_threads):
    # Note that inner_product tests use normalized input which we cannot
    # represent in int8, therefore we test only sqeuclidean metric here.
    run_cagra_hnswlib_build_search_test(
        dtype=dtype, k=k, search_params={"ef": ef, "num_threads": num_threads}
    )
