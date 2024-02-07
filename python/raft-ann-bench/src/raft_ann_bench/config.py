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

import pathlib
from typing import Any, Optional, Union

import pydantic
import yaml

from . import common as _common


class AlgoLibConfig(_common.BaseModel):
    """Configuration concerning an algorithm's library.

    # FIXME: should this be merged with AlgoConfig?

    The following is a valid specification for the `faiss_gpu_flat` algorithm:

        ```yaml
        executable: FAISS_GPU_FLAT_ANN_BENCH
        requires_gpu: true
        ```
    """

    executable: str
    requires_gpu: bool = False


class AlgoConstraints(_common.BaseModel):
    """Constraints for an algorithm.

    A constraint should be a string describing a function in a Python module.
    For example, `raft_ann_bench.has_gpu.has_gpu`.
    """

    build: Optional[str] = None
    search: Optional[str] = None


class AlgoParams(_common.BaseModel):
    """Parameters for an algorithm.

    A valid configuration looks like the following, where the build or
    search parameters must be lists:

        ```yaml
        build:
            param1: [value]
            param2: [value]
        search:
            param1: [value]
            param2: [value]
        ```
    """

    build: Optional[dict] = None
    search: Optional[dict] = None

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        if self.build:
            if not all(isinstance(v, list) for v in self.build.values()):
                raise ValueError("All build parameters should be lists")
        if self.search:
            if not all(isinstance(v, list) for v in self.search.values()):
                raise ValueError("All search parameters should be lists")


class AlgoConfig(_common.BaseModel):
    """Base class for all algorithm configurations.

    A valid configuration looks like this:

        ```yaml
        name: faiss_cpu_flat
        constraints:
            build:
            search:
        groups:
            base:
                build:
                search:
        ```
    """

    name: str
    constraints: Optional[AlgoConstraints] = pydantic.Field(
        default_factory=AlgoConstraints
    )
    groups: dict[str, AlgoParams]


class DatasetConfig(_common.BaseModel):
    """Configuration for a dataset.

    See cpp/bench/ann/src/common/conf.hpp for details.
    """

    name: str
    base_file: str
    query_file: str
    groundtruth_neighbors_file: str
    distance: str
    subset_size: Optional[int] = None
    subset_first_row: Optional[int] = None
    dtype: Optional[str] = None
    # Extra. Used for build constraints
    dims: Optional[int] = None


def load_algo_configs(
    path: Optional[Union[str, pathlib.Path]] = None,
) -> dict[str, AlgoConfig]:
    """
    Load algorithm configurations from a file or a directory.

    Args
    ----
    path (str | pathlib.Path, optional): The path to a file or a directory
        containing the configurations. If not specified, the default
        configurations will be loaded. Defaults to None.

    Returns
    -------
    dict[str, AlgoConfig]: The algorithm configurations.
    """
    path = pathlib.Path(path or _common.DEFAULT_ALGO_CONFIG_DIR)
    if path.is_file():
        with open(path) as fr:
            data = yaml.safe_load(fr.read())
            config = AlgoConfig(**data)
            return {config.name: config}
    elif path.is_dir():
        result = {}
        for pp in path.glob("*.yaml"):
            with open(pp) as fr:
                data = yaml.safe_load(fr.read())
                config = AlgoConfig(**data)
                result[config.name] = config
        return result
    return {}


def load_algo_lib_configs(
    path: Optional[Union[str, pathlib.Path]] = None,
) -> dict[str, AlgoLibConfig]:
    """Load algorithm specifications from a file.

    Args
    ----
    path (str | pathlib.Path, optional): The path to a file
        containing the algorithm library specifications. If not
        specified, the default algorithm library specifications will be
        loaded. Defaults to None.

    Returns
    -------
    dict[str, AlgoSpec]: The algorithm specifications.
    """
    path = pathlib.Path(path or _common.DEFAULT_ALGO_LIB_CONFIG_FILE)
    if path.is_file():
        with open(path) as fr:
            data = yaml.safe_load(fr.read())
            return {key: AlgoLibConfig(**value) for key, value in data.items()}
    return {}


def load_dataset_configs(
    path: Optional[Union[str, pathlib.Path]] = None,
) -> dict[str, DatasetConfig]:
    """Load dataset configurations from a file.

    Args
    ----
    path (str | pathlib.Path, optional): The path to a file
        containing the dataset configurations. If not
        specified, the default dataset configurations will be
        loaded. Defaults to None.

    Returns
    -------
    dict[str, DatasetConfig]: The dataset configurations.
    """
    path = pathlib.Path(path or _common.DEFAULT_DATASET_CONFIG_FILE)
    if path.is_file():
        with open(path) as fr:
            data = yaml.safe_load(fr.read())
            result = {}
            for ds in data:
                conf = DatasetConfig(**ds)
                result[conf.name] = conf
            return result
    return {}
