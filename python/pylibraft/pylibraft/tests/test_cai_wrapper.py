# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
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
