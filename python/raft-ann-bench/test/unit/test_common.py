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
import unittest

from raft_ann_bench.common import import_callable, lists_to_dicts


def hello_world():
    return "Hello world!"


class TestCommon(unittest.TestCase):
    def test_import_callable(self):
        callable_name = __name__ + ".hello_world"
        callable_func = import_callable(callable_name)
        assert callable_func() == "Hello world!"

    def test_lists_to_dicts(self):
        dict_of_lists = {
            "xx": [1, 2, 3],
            "yy": ["a", "b"],
            "zz": [True, False],
        }
        expected_result = [
            {"xx": 1, "yy": "a", "zz": True},
            {"xx": 1, "yy": "a", "zz": False},
            {"xx": 1, "yy": "b", "zz": True},
            {"xx": 1, "yy": "b", "zz": False},
            {"xx": 2, "yy": "a", "zz": True},
            {"xx": 2, "yy": "a", "zz": False},
            {"xx": 2, "yy": "b", "zz": True},
            {"xx": 2, "yy": "b", "zz": False},
            {"xx": 3, "yy": "a", "zz": True},
            {"xx": 3, "yy": "a", "zz": False},
            {"xx": 3, "yy": "b", "zz": True},
            {"xx": 3, "yy": "b", "zz": False},
        ]
        assert lists_to_dicts(dict_of_lists) == expected_result
