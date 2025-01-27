# Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
