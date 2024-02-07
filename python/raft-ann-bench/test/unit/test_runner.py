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

import json
import pathlib
import tempfile
import unittest
import unittest.mock

from raft_ann_bench import config
from raft_ann_bench.runner import (
    BenchBuilder,
    BenchConfig,
    BenchLogLevel,
    BenchParams,
    BenchRunner,
)

TEST_DATA_PREFIX = pathlib.Path(__file__).parent.parent / "data"


def load_bench_config() -> dict:
    """Load bench config from a file."""
    bench_config_file = TEST_DATA_PREFIX / "sift-128-euclidean.json"
    return json.loads(bench_config_file.read_text())


class TestBenchConfig(unittest.TestCase):
    def test_bench_config(self):
        BenchConfig(**load_bench_config())


class TestBenchParams(unittest.TestCase):
    def test_to_args(self):
        params = BenchParams(
            build=True,
            search=True,
            force=True,
            data_prefix="/path/to/data",
            index_prefix="/path/to/index",
            mode="latency",
            threads="4",
            raft_log_level=BenchLogLevel.debug.value,
            override_kv=["param1:value1", "param2:value2"],
            arg=["--arg1=value1", "--arg2=value2"],
        )
        args = params.to_args()
        self.assertEqual(
            set(args),
            set(
                [
                    "--build",
                    "--search",
                    "--force",
                    "--data_prefix=/path/to/data",
                    "--index_prefix=/path/to/index",
                    "--mode=latency",
                    "--threads=4",
                    "--raft_log_level=4",
                    "--override_kv=param1:value1",
                    "--override_kv=param2:value2",
                    "--arg1=value1",
                    "--arg2=value2",
                ]
            ),
        )


class TestBenchRunner(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        # Reuse an instance of BenchRunner
        self.runner = BenchRunner(
            executable_path="/path/to/executable",
            build_result_name="build_result.json",
            search_result_name="search_result.json",
            config=BenchConfig(**load_bench_config()),
            params=BenchParams(data_prefix=self.temp_dir.name),
            dry_run=False,
            temp_dir=self.temp_dir.name,
        )

    def test_write_temp_config_file(self):
        temp_config_file = self.runner.write_temp_config_file()

        self.assertTrue(temp_config_file.exists())
        temp_config = json.loads(temp_config_file.read_text())
        self.assertEqual(
            temp_config, self.runner.config.model_dump(exclude_none=True)
        )

    def test_merge_build_files(self):
        result_files = [
            TEST_DATA_PREFIX / "raft_cagra,base.json",
            TEST_DATA_PREFIX / "raft_cagra,base.json.lock",
        ]
        expected = json.loads(
            (TEST_DATA_PREFIX / "raft_cagra,base.json.merged").read_text()
        )
        merged_build_result = self.runner.merge_build_files(*result_files)
        self.assertEqual(merged_build_result, expected)

    def test_build(self):
        """TODO"""

    def test_search(self):
        """TODO"""


class TestBenchBuilder(unittest.TestCase):
    def test_generate_runners(self):
        all_algo_lib_configs = config.load_algo_lib_configs()
        all_algo_configs = config.load_algo_configs()
        all_dataset_configs = config.load_dataset_configs()

        executable = "RAFT_IVF_FLAT_ANN_BENCH"
        algo = "raft_ivf_flat"
        dataset = "sift-128-euclidean"
        k = 10
        batch_size = 1
        group = "base"
        params = BenchParams(
            # mimic the dataset path in the container
            data_prefix="/data/benchmarks/datasets"
        )
        with unittest.mock.patch.object(
            BenchBuilder, "_resolve_algo_executable", return_value=executable
        ):
            builder = BenchBuilder(
                k=k,
                batch_size=batch_size,
                params=params,
                algo_lib_config=all_algo_lib_configs[algo],
                algo_config=all_algo_configs[algo],
                dataset_config=all_dataset_configs[dataset],
            )
            runner = next(builder.generate_runners())

        expected_config = json.loads(
            (
                TEST_DATA_PREFIX
                / "sift-128-euclidean_raft_ivf_flat,base,k10,bs1_uuid.json"
            ).read_text()
        )
        expected_runner = BenchRunner(
            executable_path=executable,
            build_result_name="{},{}".format(dataset, group),
            search_result_name="{},{},k{},bs{}".format(
                dataset, group, k, batch_size
            ),
            config=BenchConfig(**expected_config),
            params=params,
        )
        self.assertEqual(runner.model_dump(), expected_runner.model_dump())
