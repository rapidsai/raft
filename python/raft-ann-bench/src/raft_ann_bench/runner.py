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

import enum
import json
import logging
import os
import pathlib
import shlex
import subprocess
import uuid
from typing import Any, Iterator, Optional, Union

import pydantic

from . import common as _common, config as _config

LOGGER = logging.getLogger(__name__)


class BenchLogLevel(enum.IntEnum):
    off = 0
    error = 1
    warn = 2
    info = 3
    debug = 4
    trace = 5


class BenchSearchBasicParamConfig(_common.BaseModel):
    k: int
    batch_size: int


class BenchIndexConfig(_common.BaseModel):
    name: str
    algo: str
    file: str
    search_result_file: Optional[str] = None
    build_param: dict
    search_params: list[dict]


class BenchConfig(_common.BaseModel):
    dataset: _config.DatasetConfig
    search_basic_param: BenchSearchBasicParamConfig
    index: list[BenchIndexConfig]


class BenchParams(_common.BaseModel):
    """Parameters for the benchmark executable.

    See cpp/bench/ann/src/common/benchmark.hpp for details.
    """

    # Custom parameters
    build: bool = False
    search: bool = False
    force: bool = False
    data_prefix: Union[str, pathlib.Path]
    index_prefix: Optional[str] = None
    mode: Optional[str] = None
    threads: Optional[str] = None
    raft_log_level: int = BenchLogLevel.info.value
    override_kv: list[str] = pydantic.Field(default_factory=list)
    # arbitrary args (e.g. Google benchmark parameters)
    extra_args: list[str] = pydantic.Field(default_factory=list)

    def to_args(self):
        """Convert the parameters to a list of command-line arguments."""
        args = []
        if self.build:
            args.append("--build")
        if self.search:
            args.append("--search")
        if self.force:
            args.append("--force")
        if self.data_prefix:
            args.append(f"--data_prefix={self.data_prefix}")
        if self.index_prefix:
            args.append(f"--index_prefix={self.index_prefix}")
        if self.mode:
            args.append(f"--mode={self.mode}")
        if self.threads:
            args.append(f"--threads={self.threads}")
        if self.raft_log_level:
            args.append(f"--raft_log_level={self.raft_log_level}")
        args += [f"--override_kv={kv}" for kv in self.override_kv]
        args += self.extra_args
        return args


class BenchBuilder(_common.BaseModel):
    k: int = 10
    batch_size: int = 1
    params: BenchParams
    algo_lib_config: _config.AlgoLibConfig
    algo_config: _config.AlgoConfig
    dataset_config: _config.DatasetConfig
    dry_run: bool = False

    def generate_runners(self) -> Iterator[BenchRunner]:
        """
        Generate BenchRunner instances for each algorithm parameter group.
        """
        self._validate_algo_lib()

        for group_name, algo_params in self.algo_config.groups.items():
            yield BenchRunner(
                executable_path=self._resolve_algo_executable(),
                build_result_name=self._make_build_result_name(group_name),
                search_result_name=self._make_search_result_name(group_name),
                config=BenchConfig(
                    dataset=self.dataset_config,
                    search_basic_param=BenchSearchBasicParamConfig(
                        k=self.k, batch_size=self.batch_size
                    ),
                    index=self._make_index_configs(group_name, algo_params),
                ),
                params=self.params,
                dry_run=self.dry_run,
            )

    def _make_index_configs(
        self, group_name: str, algo_params: _config.AlgoParams
    ) -> list[BenchIndexConfig]:
        """Get the valid parameters for the benchmark executable."""
        index_configs = []

        build_params = _common.lists_to_dicts(algo_params.build or {})
        search_params = _common.lists_to_dicts(algo_params.search or {})
        valid_build_params = [
            bparam
            for bparam in build_params
            if self._is_build_param_valid(bparam)
        ]
        for bparam in valid_build_params:
            # Construct BenchIndexConfig
            valid_search_params = [
                sparam
                for sparam in search_params
                if self._is_search_param_valid(
                    sparam, bparam, self.k, self.batch_size
                )
            ]
            index_name = self._make_index_name(
                build_param=bparam, group_name=group_name
            )
            index_file_prefix = self._make_index_file_prefix()
            index_configs.append(
                BenchIndexConfig(
                    name=index_name,
                    file=str(index_file_prefix / index_name),
                    algo=self.algo_config.name,
                    build_param=bparam,
                    search_params=valid_search_params,
                )
            )
        return index_configs

    def _resolve_algo_executable(self) -> pathlib.Path:
        """Resolve the path of the algorithm executable."""
        executable = self.algo_lib_config.executable
        path = None
        build_path = os.getenv("RAFT_HOME")
        conda_path = os.getenv("CONDA_PREFIX")
        if build_path:
            path = (
                pathlib.Path(build_path)
                / "cpp"
                / "build"
                / "release"
                / executable
            ).resolve()
        elif conda_path:
            path = (
                pathlib.Path(conda_path) / "bin" / "ann" / executable
            ).resolve()

        if not (path and path.exists()):
            raise FileNotFoundError(
                "Executable {} not found in RAFT_HOME or CONDA_PREFIX".format(
                    executable
                )
            )
        return path

    def _validate_algo_lib(self):
        """Validate the algorithm library configuration."""
        if self.algo_lib_config.requires_gpu and not _common.has_gpu():
            raise ValueError(f"GPU required for {self.algo} but not available")

        algo_lib_path = self._resolve_algo_executable()
        LOGGER.info(f"-- Using RAFT bench in {algo_lib_path}. ")

    def _is_build_param_valid(self, build_param: dict):
        constraint = self.algo_config.constraints.build
        dims = self.dataset_config.dims

        if constraint:
            if not dims:
                raise ValueError(
                    "`dims` needed for build constraints but not "
                    "specified in datasets.yaml"
                )
            validator_fn = _common.import_callable(constraint)
            return validator_fn(build_param, dims)
        return True

    def _is_search_param_valid(
        self, build_param: dict, search_param: dict, k: int, batch_size: int
    ):
        if self.algo_config.constraints.search:
            validator_fn = _common.import_callable(
                self.algo_config.constraints.search
            )
            return validator_fn(search_param, build_param, k, batch_size)
        return True

    def _make_index_name(self, build_param: dict, group_name: str) -> str:
        algo = self.algo_config.name

        index_name = algo if group_name == "base" else f"{algo}_{group_name}"
        return ".".join(
            [index_name, *(f"{kk}{vv}" for kk, vv in build_param.items())]
        )

    def _make_index_file_prefix(self) -> pathlib.Path:
        data_prefix = self.params.data_prefix
        dataset = self.dataset_config.name
        return pathlib.Path(data_prefix).resolve() / dataset / "index"

    def _make_build_result_name(self, group_name: str) -> str:
        algo = self.algo_config.name
        return ",".join([algo, group_name])

    def _make_search_result_name(self, group_name: str) -> str:
        """FIXME: consider adding the `mode` of the search to the name."""
        algo = self.algo_config.name
        return ",".join(
            [algo, group_name, f"k{self.k}", f"bs{self.batch_size}"]
        )


class BenchRunner(_common.BaseModel):
    """
    Run the benchmark executable.

    This class is responsible for calling the benchmark executable.
    The arguments reflect the parameters of the benchmark executable in
    cpp/bench/ann/src/common/benchmark.hpp

    Attributes
    ----------
    executable_path (Union[str, pathlib.Path]): The path to the benchmark
        executable.
    build_result_name (str): The name of the build result file.
    search_result_name (str): The name of the search result file.
    config (BenchConfig): The benchmark configuration.
    params (BenchParams): The CLI parameters when the executable is called.
    dry_run (bool): Whether to perform a dry run.
    temp_dir (Union[str, pathlib.Path]): The temporary directory path.
    """

    executable_path: Union[str, pathlib.Path]
    build_result_name: str
    search_result_name: str
    config: BenchConfig
    params: BenchParams
    dry_run: bool = False
    temp_dir: Union[str, pathlib.Path] = ""

    def model_post_init(self, __context: Any) -> None:
        """
        Perform post-initialization tasks.
        """
        super().model_post_init(__context)
        self.temp_dir = pathlib.Path(self.temp_dir).resolve()

    @property
    def result_prefix(self) -> pathlib.Path:
        """
        Get the result prefix for the benchmark result files.
        """
        data_prefix = pathlib.Path(self.params.data_prefix)
        legacy_result_folder = (
            data_prefix / self.config.dataset.name / "result"
        )
        legacy_result_folder.mkdir(parents=True, exist_ok=True)
        return legacy_result_folder.resolve()

    def write_temp_config_file(self) -> pathlib.Path:
        """
        Write the configuration to a temporary file.

        Returns
        -------
        pathlib.Path: The path to the temporary configuration file.
        """
        temp_config_path = pathlib.Path(self.temp_dir) / (
            "{}_{}_{}.json".format(
                self.config.dataset.name, self.search_result_name, uuid.uuid1()
            )
        )
        json_str = self.config.model_dump_json(indent=2, exclude_none=True)
        temp_config_path.write_text(json_str)
        return temp_config_path

    def merge_build_files(self, *build_files) -> dict:
        """
        Merge the build files into a single build dictionary.

        Args
        ----
        build_files: The paths to the build files.

        Returns
        -------
        dict: The merged build dictionary.
        """
        final_build_dict = {}
        for bfile in build_files:
            bfile = pathlib.Path(bfile)
            if bfile.exists():
                build_dict = json.loads(bfile.read_text())
                if not final_build_dict:
                    final_build_dict = build_dict
                else:
                    final_build_dict["benchmarks"] = [
                        *final_build_dict.get("benchmarks", []),
                        *build_dict.get("benchmarks", []),
                    ]
        return final_build_dict

    def build(self):
        """
        Run the executable with build params.
        """
        temp_config_path = self.write_temp_config_file()

        result_path = (
            self.result_prefix / "build" / f"{self.build_result_name}.json"
        )
        result_path.parent.mkdir(parents=True, exist_ok=True)
        temp_result_path = result_path.parent / f"{result_path.name}.lock"

        params = self.params.model_copy()
        params.build = True
        params.search = False
        params.extra_args += [
            f"--{kk}={vv}"
            for kk, vv in {
                "benchmark_out_format": "json",
                "benchmark_counters_tabular": "true",
                "benchmark_out": str(temp_result_path),
            }.items()
        ]
        cmd = list(
            map(
                str,
                [self.executable_path, *params.to_args(), temp_config_path],
            )
        )
        try:
            if self.dry_run:
                cmd_str = shlex.join(cmd)
                print(cmd_str)
                LOGGER.info(
                    "Benchmark command for {}:\n{}".format(
                        self.build_result_name, cmd_str
                    )
                )
            else:
                try:
                    subprocess.run(cmd, check=True)
                    merged_build_dict = self.merge_build_files(
                        result_path, temp_result_path
                    )
                    with open(result_path, "w") as fw:
                        json_str = json.dumps(merged_build_dict, indent=2)
                        fw.write(json_str)
                except Exception as e:
                    LOGGER.error("Error occurred running benchmark: %s" % e)
                    raise e
        finally:
            temp_result_path.unlink(missing_ok=True)
            temp_config_path.unlink()

    def search(self):
        """
        Run the executable with search params.
        """
        temp_config_path = self.write_temp_config_file()

        result_path = (
            self.result_prefix / "search" / f"{self.search_result_name}.json"
        )
        result_path.parent.mkdir(parents=True, exist_ok=True)

        params = self.params.model_copy()
        params.build = False
        params.search = True
        params.extra_args += [
            f"--{kk}={vv}"
            for kk, vv in {
                "benchmark_out_format": "json",
                "benchmark_counters_tabular": "true",
                "benchmark_min_warmup_time": "1",
                "benchmark_out": str(result_path),
            }.items()
        ]
        cmd = list(
            map(
                str,
                [self.executable_path, *params.to_args(), temp_config_path],
            )
        )
        try:
            if self.dry_run:
                cmd_str = shlex.join(cmd)
                print(cmd_str)
                LOGGER.info(
                    "Benchmark command for {}:\n{}".format(
                        self.search_result_name, cmd_str
                    )
                )
            else:
                try:
                    subprocess.run(cmd, check=True)
                except Exception as e:
                    LOGGER.error("Error occurred running benchmark: %s" % e)
                    raise e
        finally:
            temp_config_path.unlink()
