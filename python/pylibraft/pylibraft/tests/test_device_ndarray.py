# Copyright (c) 2022, NVIDIA CORPORATION.
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

from pylibraft.common import device_ndarray


@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_basic_attributes(order, dtype):

    a = np.random.random((500, 2)).astype(dtype)

    if order == "C":
        a = np.ascontiguousarray(a)
    else:
        a = np.asfortranarray(a)

    db = device_ndarray(a)
    db_host = db.copy_to_host()

    assert a.shape == db.shape
    assert a.dtype == db.dtype
    assert a.data.f_contiguous == db.f_contiguous
    assert a.data.f_contiguous == db_host.data.f_contiguous
    assert a.data.c_contiguous == db.c_contiguous
    assert a.data.c_contiguous == db_host.data.c_contiguous
    np.testing.assert_array_equal(a.tolist(), db_host.tolist())


@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_empty(order, dtype):

    a = np.random.random((500, 2)).astype(dtype)
    if order == "C":
        a = np.ascontiguousarray(a)
    else:
        a = np.asfortranarray(a)

    db = device_ndarray.empty(a.shape, dtype=dtype, order=order)
    db_host = db.copy_to_host()

    assert a.shape == db.shape
    assert a.dtype == db.dtype
    assert a.data.f_contiguous == db.f_contiguous
    assert a.data.f_contiguous == db_host.data.f_contiguous
    assert a.data.c_contiguous == db.c_contiguous
    assert a.data.c_contiguous == db_host.data.c_contiguous
