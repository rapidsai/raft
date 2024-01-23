# Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

import numpy as np
import pytest
from scipy.sparse import csr_array

from pylibraft.common import DeviceResources, Stream
from pylibraft.neighbors.brute_force import eps_neighbors as eps_neighbors_bf
from pylibraft.neighbors.rbc import (
    build_rbc_index,
    eps_neighbors as eps_neighbors_rbc,
)


def test_bf_eps_neighbors_check_col_major_inputs():
    # make sure that we get an exception if passed col-major inputs,
    # instead of returning incorrect results
    cp = pytest.importorskip("cupy")
    n_index_rows, n_query_rows, n_cols = 128, 16, 32
    eps = 0.02
    index = cp.random.random_sample((n_index_rows, n_cols), dtype="float32")
    queries = cp.random.random_sample((n_query_rows, n_cols), dtype="float32")

    with pytest.raises(ValueError):
        eps_neighbors_bf(cp.asarray(index, order="F"), queries, eps)

    with pytest.raises(ValueError):
        eps_neighbors_bf(index, cp.asarray(queries, order="F"), eps)

    # shouldn't throw an exception with c-contiguous inputs
    eps_neighbors_bf(index, queries, eps)


def test_rbc_eps_neighbors_check_col_major_inputs():
    # make sure that we get an exception if passed col-major inputs,
    # instead of returning incorrect results
    cp = pytest.importorskip("cupy")
    n_index_rows, n_query_rows, n_cols = 128, 16, 32
    eps = 0.02
    index = cp.random.random_sample((n_index_rows, n_cols), dtype="float32")
    queries = cp.random.random_sample((n_query_rows, n_cols), dtype="float32")

    with pytest.raises(ValueError):
        build_rbc_index(cp.asarray(index, order="F"))

    rbc_index = build_rbc_index(index)

    with pytest.raises(ValueError):
        eps_neighbors_rbc(rbc_index, cp.asarray(queries, order="F"), eps)

    eps_neighbors_rbc(rbc_index, queries, eps)


@pytest.mark.parametrize("n_index_rows", [32, 100, 1000])
@pytest.mark.parametrize("n_query_rows", [32, 100, 1000])
@pytest.mark.parametrize("n_cols", [2, 3, 40, 100])
def test_eps_neighbors(n_index_rows, n_query_rows, n_cols):
    s2 = Stream()
    handle = DeviceResources(stream=s2)

    cp = pytest.importorskip("cupy")
    eps = 0.02
    index = cp.random.random_sample((n_index_rows, n_cols), dtype="float32")
    queries = cp.random.random_sample((n_query_rows, n_cols), dtype="float32")

    # brute force
    adj_bf, vd_bf = eps_neighbors_bf(index, queries, eps, handle=handle)
    adj_bf = cp.asarray(adj_bf)
    vd_bf = cp.asarray(vd_bf)

    rbc_index = build_rbc_index(index, handle=handle)
    adj_rbc_ia, adj_rbc_ja, vd_rbc = eps_neighbors_rbc(
        rbc_index, queries, eps, handle=handle
    )
    adj_rbc_ia = cp.asarray(adj_rbc_ia)
    adj_rbc_ja = cp.asarray(adj_rbc_ja)
    vd_rbc = cp.asarray(vd_rbc)

    np.testing.assert_array_equal(vd_bf.get(), vd_rbc.get())

    adj_rbc = csr_array(
        (
            np.ones(adj_rbc_ia.get()[n_query_rows]),
            adj_rbc_ja.get(),
            adj_rbc_ia.get(),
        ),
        shape=(n_query_rows, n_index_rows),
    ).toarray()
    np.testing.assert_array_equal(adj_bf.get(), adj_rbc)
