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

import importlib
import importlib.resources
import itertools
import logging
import os
import pathlib
from typing import Callable

import pydantic
import rich.logging

_MODULE = __name__.split(".")[0]
DEFAULT_CONFIG_DIR = importlib.resources.files(_MODULE) / "data" / "conf"
DEFAULT_ALGO_CONFIG_DIR = DEFAULT_CONFIG_DIR / "algo"
DEFAULT_DATSET_CONFIG_DIR = DEFAULT_CONFIG_DIR / "dataset"
DEFAULT_ALGO_LIB_CONFIG_FILE = DEFAULT_CONFIG_DIR / "algos.yaml"
DEFAULT_DATASET_CONFIG_FILE = DEFAULT_CONFIG_DIR / "datasets.yaml"


class BaseModel(pydantic.BaseModel):
    """Use the same Pydantic BaseModel configuration for all models."""

    model_config = pydantic.ConfigDict(extra="forbid")


def get_data_prefix() -> pathlib.Path:
    """
    Get the default dataset path.
    """
    path = os.getenv("RAPIDS_DATASET_ROOT_DIR")
    if not path:
        path = pathlib.Path() / "datasets"
    return pathlib.Path(path).resolve()


def has_gpu():
    """
    Check if the system has a GPU available.
    """
    try:
        import rmm  # noqa: F401

        return True
    except ImportError:
        return False


def reset_logger(level: int = logging.WARN):
    """
    Reset the logger for the current module with the specified log level.

    This is a fix to account for `typer` modifying the root logger
    when `typer` is imported.

    Args
    ----
    level (int): The log level to set for the logger. Defaults to
    logging.WARN.
    """
    module_name = __name__.rsplit(".", 1)[0]
    module_logger = logging.getLogger(module_name)
    module_logger.setLevel(level)
    module_logger.handlers.clear()
    module_logger.addHandler(rich.logging.RichHandler())


def import_callable(name: str) -> Callable:
    """
    Import a callable from a module.

    Args
    ----
    name (str): The name of the callable to import.
    """
    module_name, callable_name = name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    callable = getattr(module, callable_name)
    if isinstance(callable, Callable):
        return callable
    else:
        raise ValueError(f"The imported object {name} is not callable.")


def lists_to_dicts(dict_of_lists: dict[str, list]) -> list[dict]:
    """Convert a dictionary of lists to a list of dictionaries."""
    dict_keys = dict_of_lists.keys()
    dict_value_tuples = itertools.product(*dict_of_lists.values())

    return [
        {kk: vv for kk, vv in zip(dict_keys, vtuple)}
        for vtuple in dict_value_tuples
    ]
