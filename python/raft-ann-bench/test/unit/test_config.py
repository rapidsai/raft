# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
from __future__ import annotations

import unittest

from raft_ann_bench.config import (
    load_algo_configs,
    load_algo_lib_configs,
    load_dataset_configs,
)


class TestConfig(unittest.TestCase):
    def test_algo_config(self):
        valid = load_algo_configs()
        self.assertTrue(valid)
        invalid = load_algo_configs("dummy.yaml")
        self.assertFalse(invalid)

    def test_algo_lib_config(self):
        valid = load_algo_lib_configs()
        self.assertTrue(valid)
        invalid = load_algo_lib_configs("dummy.yaml")
        self.assertFalse(invalid)

    def test_dataset_config(self):
        valid = load_dataset_configs()
        self.assertTrue(valid)
        invalid = load_dataset_configs("dummy.yaml")
        self.assertFalse(invalid)


if __name__ == "__main__":
    unittest.main()
