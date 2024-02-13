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
from typing import Optional

import typer

from . import common as _common, config as _config, runner as _runner

app = typer.Typer(pretty_exceptions_show_locals=False)
runner_app = typer.Typer(pretty_exceptions_show_locals=False)
app.add_typer(runner_app, name="run")

# Remap the enum integer values to strings
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
    dataset: str = typer.Option(..., help="Name of the dataset."),
    algorithms: str = typer.Option(
        ..., help="Comma-separated list of algorithm names."
    ),
    k: int = typer.Option(10, help="Number of nearest neighbors to retrieve."),
    batch_size: int = typer.Option(
        1, help="Number of queries to process in each batch."
    ),
    algorithm_config: Optional[str] = typer.Option(
        None, help="Path to an additional algorithm configuration file."
    ),
    algorithm_library_config: Optional[str] = typer.Option(
        None,
        help="Path to an additional algorithm library configuration file.",
    ),
    dataset_config: Optional[str] = typer.Option(
        None, help="Path to an additional dataset configuration file."
    ),
    dry_run: bool = typer.Option(False, help="If True, performs a dry run."),
):
    """
    Run the benchmark executable.

    Notes that this main command will set up the necessary configurations
    for running the subcommands `build` and `search`.
    """

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
    data_prefix: Optional[str] = typer.Option(
        None, help="The prefix for the data files."
    ),
    force: bool = typer.Option(
        False, help="Flag indicating whether to force overwrite."
    ),
    raft_log_level: BenchLogLevelStr = typer.Option(  # type: ignore
        BenchLogLevelStr.info.value, help="The log level for the raft library."
    ),
    override_kv: list[str] = typer.Option(
        None, help="The override key-value pairs."
    ),
):
    """Pass on build parameters to the ANN executable.

    In addition to the parameters specified here, the command also accepts
    arbitrary key-value pairs as extra arguments. These key-value pairs will
    be passed to the executable as is. See
    `cpp/bench/ann/src/common/benchmark.hpp` for details.

    """
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
    data_prefix: Optional[str] = typer.Option(
        None, help="The prefix for the data files."
    ),
    index_prefix: Optional[str] = typer.Option(
        None, help="The prefix for the index files."
    ),
    mode: BenchSearchMode = typer.Option(None, help="The search mode."),
    threads: str = typer.Option(None, help="The number of threads to use."),
    force: bool = typer.Option(
        False, help="Flag indicating whether to force overwrite."
    ),
    raft_log_level: BenchLogLevelStr = typer.Option(  # type: ignore
        BenchLogLevelStr.info.value, help="The log level for the raft library."
    ),
    override_kv: list[str] = typer.Option(
        None, help="The override key-value pairs."
    ),
):
    """Pass on search parameters to the ANN executable.

    In addition to the parameters specified here, the command also accepts
    arbitrary key-value pairs as extra arguments. These key-value pairs will
    be passed to the executable as is. See
    `cpp/bench/ann/src/common/benchmark.hpp` for details.
    """
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
