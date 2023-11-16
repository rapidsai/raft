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

from pylibraft.common import cai_wrapper, device_ndarray


@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("shape", [(10, 5)])
def test_basic_accessors(order, dtype, shape):

    a = np.random.random(shape).astype(dtype)

    if order == "C":
        a = np.ascontiguousarray(a)
    else:
        a = np.asfortranarray(a)

    db = device_ndarray(a)
    cai_wrap = cai_wrapper(db)

    assert cai_wrap.dtype == dtype
    assert cai_wrap.shape == shape
    assert cai_wrap.c_contiguous == (order == "C")
    assert cai_wrap.f_contiguous == (order == "F")
