# `raft-ann-bench`

**Usage**:

```console
$ raft-ann-bench [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--verbose / --no-verbose`: [default: no-verbose]
* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `run`: Main function for running the benchmarking...

## `raft-ann-bench run`

Main function for running the benchmarking executable.

Notes that this main command will set up the necessary configurations
for running the subcommands `build` and `search`.

**Usage**:

```console
$ raft-ann-bench run [OPTIONS] [DATASET] [ALGORITHMS] COMMAND [ARGS]...
```

**Arguments**:

* `[DATASET]`: Name of the dataset.
* `[ALGORITHMS]`: Comma-separated list of algorithm names.

**Options**:

* `--k INTEGER`: Number of nearest neighbors to retrieve.  [default: 10]
* `--batch-size INTEGER`: Number of queries to process in each batch.  [default: 1]
* `--algorithm-config TEXT`: Path to an additional algorithm configuration file.
* `--algorithm-library-config TEXT`: Path to an additional algorithm library configuration file.
* `--dataset-config TEXT`: Path to an additional dataset configuration file.
* `--dry-run / --no-dry-run`: If True, performs a dry run.  [default: no-dry-run]
* `--help`: Show this message and exit.

**Commands**:

* `build`: Pass on build parameters to the ANN...
* `search`: Pass on search parameters to the ANN...

### `raft-ann-bench run build`

Pass on build parameters to the ANN executable.

In addition to the parameters specified here, the command also accepts
arbitrary key-value pairs as extra arguments. These key-value pairs will
be passed to the executable as is. See
`cpp/bench/ann/src/common/benchmark.hpp` for details.

**Usage**:

```console
$ raft-ann-bench run build [OPTIONS]
```

**Options**:

* `--data-prefix TEXT`: The prefix for the data files.
* `--force / --no-force`: Flag indicating whether to force overwrite.  [default: no-force]
* `--raft-log-level [off|error|warn|info|debug|trace]`: The log level for the raft library.  [default: info]
* `--override-kv TEXT`: The override key-value pairs.
* `--help`: Show this message and exit.

### `raft-ann-bench run search`

Pass on search parameters to the ANN executable.

In addition to the parameters specified here, the command also accepts
arbitrary key-value pairs as extra arguments. These key-value pairs will
be passed to the executable as is. See
`cpp/bench/ann/src/common/benchmark.hpp` for details.

**Usage**:

```console
$ raft-ann-bench run search [OPTIONS]
```

**Options**:

* `--data-prefix TEXT`: The prefix for the data files.
* `--index-prefix TEXT`: The prefix for the index files.
* `--mode [latency|throughput]`: The search mode.
* `--threads TEXT`: The number of threads to use.
* `--force / --no-force`: Flag indicating whether to force overwrite.  [default: no-force]
* `--raft-log-level [off|error|warn|info|debug|trace]`: The log level for the raft library.  [default: info]
* `--override-kv TEXT`: The override key-value pairs.
* `--help`: Show this message and exit.
