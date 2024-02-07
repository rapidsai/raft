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
import logging
from typing import Annotated, List, Optional

import typer

from . import common as _common, config as _config, runner as _runner

app = typer.Typer(pretty_exceptions_show_locals=False)
runner_app = typer.Typer(pretty_exceptions_show_locals=False)
app.add_typer(runner_app, name="run")


BenchLogLevelStr = enum.Enum(
    "BenchLogLevelStr",
    {kk: kk for kk in _runner.BenchLogLevel.__members__.keys()},
)
BenchSearchMode = enum.Enum(
    "BenchSearchMode", dict(latency="latency", throughput="throughput")
)


class State(_common.BaseModel):
    verbose: bool = False
    runner: Optional[RunnerState] = None


class RunnerState(_common.BaseModel):
    algorithms: list[str]
    dataset: str
    k: int
    batch_size: int
    algo_lib_configs: dict[str, _config.AlgoLibConfig]
    algo_configs: dict[str, _config.AlgoConfig]
    dataset_config: _config.DatasetConfig
    dry_run: bool


@app.callback()
def main(ctx: typer.Context, verbose: bool = False):
    if verbose:
        _common.reset_logger(level=logging.DEBUG)

    ctx.obj = State(verbose=verbose)


@runner_app.callback()
def runner_main(
    ctx: typer.Context,
    algorithms: str = typer.Option(),
    dataset: str = typer.Option(),
    k: int = 10,
    batch_size: int = 1,
    algorithm_config: Optional[str] = None,
    algorithm_library_config: Optional[str] = None,
    dataset_config: Optional[str] = None,
    dry_run: bool = False,
):
    algorithms = algorithms.split(",")

    all_dataset_configs = {
        **_config.load_dataset_configs(),
        **(
            _config.load_dataset_configs(dataset_config)
            if dataset_config
            else {}
        ),
    }

    all_algo_configs = {
        **_config.load_algo_configs(),
        **(
            _config.load_algo_configs(algorithm_config)
            if algorithm_config
            else {}
        ),
    }

    all_algo_lib_configs = {
        **_config.load_algo_lib_configs(),
        **(
            _config.load_algo_lib_configs(algorithm_library_config)
            if algorithm_library_config
            else {}
        ),
    }
    # Verify that the necessary configs are available
    for algo in algorithms:
        if algo not in all_algo_lib_configs:
            raise ValueError(
                f"Algorithm library config not found for '{algo}'."
            )
        if algo not in all_algo_configs:
            raise ValueError(f"Algorithm config not found for '{algo}'.")
    if dataset not in all_dataset_configs:
        raise ValueError(f"Dataset config not found for '{dataset}'.")

    ctx.obj.runner = RunnerState(
        algorithms=algorithms,
        dataset=dataset,
        k=k,
        batch_size=batch_size,
        algo_lib_configs={
            algo: all_algo_lib_configs[algo] for algo in algorithms
        },
        algo_configs={algo: all_algo_configs[algo] for algo in algorithms},
        dataset_config=all_dataset_configs[dataset],
        dry_run=dry_run,
    )


@runner_app.command(
    name="build",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
)
def runner_build(
    ctx: typer.Context,
    data_prefix: Optional[str] = None,
    force: bool = False,
    raft_log_level: BenchLogLevelStr = BenchLogLevelStr.info.value,
    override_kv: Annotated[Optional[List[str]], typer.Option()] = None,
):
    """Build parameters passed on to the ANN executable."""
    raft_log_level = getattr(_runner.BenchLogLevel, raft_log_level.value)
    data_prefix = data_prefix or _common.get_data_prefix()

    params = _runner.BenchParams(
        data_prefix=data_prefix,
        force=force,
        raft_log_level=raft_log_level.value,
        override_kv=override_kv,
        extra_args=ctx.args,
    )
    state = RunnerState.model_validate(ctx.obj.runner)
    for algo in state.algorithms:
        builder = _runner.BenchBuilder(
            k=state.k,
            batch_size=state.batch_size,
            params=params,
            algo_lib_config=state.algo_lib_configs.get(algo),
            algo_config=state.algo_configs.get(algo),
            dataset_config=state.dataset_config,
            dry_run=state.dry_run,
        )
        for runner in builder.generate_runners():
            runner.build()


@runner_app.command(
    name="search",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
    },
)
def runner_search(
    ctx: typer.Context,
    data_prefix: Optional[str] = None,
    index_prefix: Optional[str] = None,
    mode: Optional[BenchSearchMode] = None,
    threads: Optional[str] = None,
    force: bool = False,
    raft_log_level: BenchLogLevelStr = BenchLogLevelStr.info.value,
    override_kv: Annotated[Optional[List[str]], typer.Option()] = None,
):
    """Search parameters passed on to the ANN executable."""
    raft_log_level = getattr(_runner.BenchLogLevel, raft_log_level.value)
    data_prefix = data_prefix or _common.get_data_prefix()

    params = _runner.BenchParams(
        data_prefix=data_prefix,
        index_prefix=index_prefix,
        mode=mode,
        threads=threads,
        force=force,
        raft_log_level=raft_log_level.value,
        override_kv=override_kv,
        extra_args=ctx.args,
    )
    state = RunnerState.model_validate(ctx.obj.runner)
    for algo in state.algorithms:
        builder = _runner.BenchBuilder(
            k=state.k,
            batch_size=state.batch_size,
            params=params,
            algo_lib_config=state.algo_lib_configs.get(algo),
            algo_config=state.algo_configs.get(algo),
            dataset_config=state.dataset_config,
            dry_run=state.dry_run,
        )
        for runner in builder.generate_runners():
            runner.search()


if __name__ == "__main__":
    app()
