# SPDX-FileCopyrightText: Copyright (c) 2020-2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import sys

import pytest

try:
    import raft_dask
except ImportError:
    print("Skipping RAFT tests")
    pytestmart = pytest.mark.skip

pytestmark = pytest.mark.skipif(
    "raft_dask" not in sys.argv, reason="marker to allow integration of RAFT"
)


def test_raft():
    assert raft_dask.raft_include_test()
