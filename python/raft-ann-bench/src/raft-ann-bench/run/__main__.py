#
# Copyright (c) 2023, NVIDIA CORPORATION.
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

import argparse
import itertools
import json
import os
import subprocess
import sys
import uuid
import warnings
from importlib import import_module

import yaml

log_levels = {
    "off": 0,
    "error": 1,
    "warn": 2,
    "info": 3,
    "debug": 4,
    "trace": 5,
}


def parse_log_level(level_str):
    if level_str not in log_levels:
        raise ValueError("Invalid log level: %s" % level_str)
    return log_levels[level_str.lower()]


def positive_int(input_str: str) -> int:
    try:
        i = int(input_str)
        if i < 1:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"{input_str} is not a positive integer"
        )

    return i


def merge_build_files(build_dir, build_file, temp_build_file):

    build_dict = {}

    # If build file exists, read it
    build_json_path = os.path.join(build_dir, build_file)
    tmp_build_json_path = os.path.join(build_dir, temp_build_file)
    if os.path.isfile(build_json_path):
        try:
            with open(build_json_path, "r") as f:
                build_dict = json.load(f)
        except Exception as e:
            print(
                "Error loading existing build file: %s (%s)"
                % (build_json_path, e)
            )

    temp_build_dict = {}
    if os.path.isfile(tmp_build_json_path):
        with open(tmp_build_json_path, "r") as f:
            temp_build_dict = json.load(f)
    else:
        raise ValueError("Temp build file not found: %s" % tmp_build_json_path)

    tmp_benchmarks = (
        temp_build_dict["benchmarks"]
        if "benchmarks" in temp_build_dict
        else {}
    )
    benchmarks = build_dict["benchmarks"] if "benchmarks" in build_dict else {}

    # If the build time is absolute 0 then an error occurred
    final_bench_dict = {}
    for b in benchmarks:
        if b["real_time"] > 0:
            final_bench_dict[b["name"]] = b

    for tmp_bench in tmp_benchmarks:
        if tmp_bench["real_time"] > 0:
            final_bench_dict[tmp_bench["name"]] = tmp_bench

    temp_build_dict["benchmarks"] = [v for k, v in final_bench_dict.items()]
    with open(build_json_path, "w") as f:
        json_str = json.dumps(temp_build_dict, indent=2)
        f.write(json_str)


def validate_algorithm(algos_conf, algo, gpu_present):
    algos_conf_keys = set(algos_conf.keys())
    if gpu_present:
        return algo in algos_conf_keys
    else:
        return (
            algo in algos_conf_keys
            and algos_conf[algo]["requires_gpu"] is False
        )


def find_executable(algos_conf, algo, group, k, batch_size):
    executable = algos_conf[algo]["executable"]

    file_name = (f"{algo},{group}", f"{algo},{group},k{k},bs{batch_size}")

    build_path = os.getenv("RAFT_HOME")
    if build_path is not None:
        build_path = os.path.join(
            build_path, "cpp", "build", "release", executable
        )
        if os.path.exists(build_path):
            print(f"-- Using RAFT bench from repository in {build_path}. ")
            return (executable, build_path, file_name)

    # if there is no build folder present, we look in the conda environment
    conda_path = os.getenv("CONDA_PREFIX")
    if conda_path is not None:
        conda_path = os.path.join(conda_path, "bin", "ann", executable)
        if os.path.exists(conda_path):
            print("-- Using RAFT bench found in conda environment. ")
            return (executable, conda_path, file_name)

    else:
        raise FileNotFoundError(executable)


def run_build_and_search(
    conf_file,
    conf_filename,
    conf_filedir,
    executables_to_run,
    dataset_path,
    force,
    build,
    search,
    dry_run,
    k,
    batch_size,
    search_threads,
    mode="throughput",
    raft_log_level="info",
):
    for (
        executable,
        ann_executable_path,
        output_filename,
    ) in executables_to_run.keys():
        # Need to write temporary configuration
        temp_conf_filename = (
            f"{conf_filename}_{output_filename[1]}_{uuid.uuid1()}.json"
        )
        with open(temp_conf_filename, "w") as f:
            temp_conf = dict()
            temp_conf["dataset"] = conf_file["dataset"]
            temp_conf["search_basic_param"] = conf_file["search_basic_param"]
            temp_conf["index"] = executables_to_run[
                (executable, ann_executable_path, output_filename)
            ]["index"]
            json_str = json.dumps(temp_conf, indent=2)
            f.write(json_str)

        legacy_result_folder = os.path.join(
            dataset_path, conf_file["dataset"]["name"], "result"
        )
        os.makedirs(legacy_result_folder, exist_ok=True)
        if build:
            build_folder = os.path.join(legacy_result_folder, "build")
            os.makedirs(build_folder, exist_ok=True)
            build_file = f"{output_filename[0]}.json"
            temp_build_file = f"{build_file}.lock"
            cmd = [
                ann_executable_path,
                "--build",
                "--data_prefix=" + dataset_path,
                "--benchmark_out_format=json",
                "--benchmark_counters_tabular=true",
                "--benchmark_out="
                + f"{os.path.join(build_folder, temp_build_file)}",
                "--raft_log_level=" + f"{parse_log_level(raft_log_level)}",
            ]
            if force:
                cmd = cmd + ["--force"]
            cmd = cmd + [temp_conf_filename]

            if dry_run:
                print(
                    "Benchmark command for %s:\n%s\n"
                    % (output_filename[0], " ".join(cmd))
                )
            else:
                try:
                    subprocess.run(cmd, check=True)
                    merge_build_files(
                        build_folder, build_file, temp_build_file
                    )
                except Exception as e:
                    print("Error occurred running benchmark: %s" % e)
                finally:
                    os.remove(os.path.join(build_folder, temp_build_file))
                    if not search:
                        os.remove(temp_conf_filename)

        if search:
            search_folder = os.path.join(legacy_result_folder, "search")
            os.makedirs(search_folder, exist_ok=True)
            search_file = f"{output_filename[1]}.json"
            cmd = [
                ann_executable_path,
                "--search",
                "--data_prefix=" + dataset_path,
                "--benchmark_counters_tabular=true",
                "--override_kv=k:%s" % k,
                "--override_kv=n_queries:%s" % batch_size,
                "--benchmark_min_warmup_time=1",
                "--benchmark_out_format=json",
                "--mode=%s" % mode,
                "--benchmark_out="
                + f"{os.path.join(search_folder, search_file)}",
                "--raft_log_level=" + f"{parse_log_level(raft_log_level)}",
            ]
            if force:
                cmd = cmd + ["--force"]

            if search_threads:
                cmd = cmd + ["--threads=%s" % search_threads]

            cmd = cmd + [temp_conf_filename]
            if dry_run:
                print(
                    "Benchmark command for %s:\n%s\n"
                    % (output_filename[1], " ".join(cmd))
                )
            else:
                try:
                    subprocess.run(cmd, check=True)
                except Exception as e:
                    print("Error occurred running benchmark: %s" % e)
                finally:
                    os.remove(temp_conf_filename)


def main():
    scripts_path = os.path.dirname(os.path.realpath(__file__))
    call_path = os.getcwd()

    # Read list of allowed algorithms
    try:
        import rmm  # noqa: F401

        gpu_present = True
    except ImportError:
        gpu_present = False

    with open(f"{scripts_path}/algos.yaml", "r") as f:
        algos_yaml = yaml.safe_load(f)

    if "RAPIDS_DATASET_ROOT_DIR" in os.environ:
        default_dataset_path = os.getenv("RAPIDS_DATASET_ROOT_DIR")
    else:
        default_dataset_path = os.path.join(call_path, "datasets/")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--subset-size",
        type=positive_int,
        help="the number of subset rows of the dataset to build the index",
    )
    parser.add_argument(
        "-k",
        "--count",
        default=10,
        type=positive_int,
        help="the number of nearest neighbors to search for",
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        default=10000,
        type=positive_int,
        help="number of query vectors to use in each query trial",
    )
    parser.add_argument(
        "--dataset-configuration",
        help="path to YAML configuration file for datasets",
    )
    parser.add_argument(
        "--configuration",
        help="path to YAML configuration file or directory for algorithms\
              Any run groups found in the specified file/directory will \
              automatically override groups of the same name present in the \
              default configurations, including `base`",
    )
    parser.add_argument(
        "--dataset",
        help="name of dataset",
        default="glove-100-inner",
    )
    parser.add_argument(
        "--dataset-path",
        help="path to dataset folder, by default will look in "
        "RAPIDS_DATASET_ROOT_DIR if defined, otherwise a datasets "
        "subdirectory from the calling directory",
        default=default_dataset_path,
    )
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--search", action="store_true")
    parser.add_argument(
        "--algorithms",
        help="run only comma separated list of named \
              algorithms. If parameters `groups` and `algo-groups \
              are both undefined, then group `base` is run by default",
        default=None,
    )
    parser.add_argument(
        "--groups",
        help="run only comma separated groups of parameters",
        default="base",
    )
    parser.add_argument(
        "--algo-groups",
        help='add comma separated <algorithm>.<group> to run. \
              Example usage: "--algo-groups=raft_cagra.large,hnswlib.large"',
    )
    parser.add_argument(
        "-f",
        "--force",
        help="re-run algorithms even if their results \
                              already exist",
        action="store_true",
    )

    parser.add_argument(
        "-m",
        "--search-mode",
        help="run search in 'latency' (measure individual batches) or "
        "'throughput' (pipeline batches and measure end-to-end) mode",
        default="latency",
    )

    parser.add_argument(
        "-t",
        "--search-threads",
        help="specify the number threads to use for throughput benchmark."
        " Single value or a pair of min and max separated by ':'. "
        "Example: --search-threads=1:4. Power of 2 values between 'min' "
        "and 'max' will be used. If only 'min' is specified, then a "
        "single test is run with 'min' threads. By default min=1, "
        "max=<num hyper threads>.",
        default=None,
    )

    parser.add_argument(
        "-r",
        "--dry-run",
        help="dry-run mode will convert the yaml config for the specified "
        "algorithms and datasets to the json format that's consumed "
        "by the lower-level c++ binaries and then print the command "
        "to run execute the benchmarks but will not actually execute "
        "the command.",
        action="store_true",
    )
    parser.add_argument(
        "--raft-log-level",
        help="Log level, possible values are "
        "[off, error, warn, info, debug, trace]. "
        "Default: 'info'. Note that 'debug' or more detailed "
        "logging level requires that the library is compiled with "
        "-DRAFT_ACTIVE_LEVEL=<L> where <L> >= <requested log level>",
        default="info",
    )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    # If both build and search are not provided,
    # run both
    if not args.build and not args.search:
        build = True
        search = True
    else:
        build = args.build
        search = args.search

    dry_run = args.dry_run

    mode = args.search_mode
    k = args.count
    batch_size = args.batch_size

    # Read configuration file associated to datasets
    if args.dataset_configuration:
        dataset_conf_f = args.dataset_configuration
    else:
        dataset_conf_f = os.path.join(scripts_path, "conf", "datasets.yaml")
    with open(dataset_conf_f, "r") as f:
        dataset_conf_all = yaml.safe_load(f)

    dataset_conf = None
    for dataset in dataset_conf_all:
        if args.dataset == dataset["name"]:
            dataset_conf = dataset
            break
    if not dataset_conf:
        raise ValueError("Could not find a dataset configuration")

    conf_file = dict()
    conf_file["dataset"] = dataset_conf
    if args.subset_size:
        conf_file["dataset"]["subset_size"] = args.subset_size

    conf_file["search_basic_param"] = {}
    conf_file["search_basic_param"]["k"] = k
    conf_file["search_basic_param"]["batch_size"] = batch_size

    algos_conf_fs = os.listdir(os.path.join(scripts_path, "conf", "algos"))
    algos_conf_fs = [
        os.path.join(scripts_path, "conf", "algos", f)
        for f in algos_conf_fs
        if ".json" not in f
    ]
    conf_filedir = os.path.join(scripts_path, "conf", "algos")
    if args.configuration:
        if os.path.isdir(args.configuration):
            conf_filedir = args.configuration
            algos_conf_fs = algos_conf_fs + [
                os.path.join(args.configuration, f)
                for f in os.listdir(args.configuration)
                if ".json" not in f
            ]
        elif os.path.isfile(args.configuration):
            conf_filedir = os.path.normpath(args.configuration).split(os.sep)
            conf_filedir = os.path.join(*conf_filedir[:-1])
            algos_conf_fs = algos_conf_fs + [args.configuration]

    filter_algos = True if args.algorithms else False
    if filter_algos:
        allowed_algos = args.algorithms.split(",")
    named_groups = args.groups.split(",")
    filter_algo_groups = True if args.algo_groups else False
    allowed_algo_groups = None
    if filter_algo_groups:
        allowed_algo_groups = [
            algo_group.split(".") for algo_group in args.algo_groups.split(",")
        ]
        allowed_algo_groups = list(zip(*allowed_algo_groups))
    algos_conf = dict()
    for algo_f in algos_conf_fs:
        with open(algo_f, "r") as f:
            try:
                algo = yaml.safe_load(f)
            except Exception as e:
                warnings.warn(
                    f"Could not load YAML config {algo_f} due to "
                    + e.with_traceback()
                )
                continue
            insert_algo = True
            insert_algo_group = False
            if filter_algos:
                if algo["name"] not in allowed_algos:
                    insert_algo = False
            if filter_algo_groups:
                if algo["name"] in allowed_algo_groups[0]:
                    insert_algo_group = True

            def add_algo_group(group_list):
                if algo["name"] not in algos_conf:
                    algos_conf[algo["name"]] = {"groups": {}}
                for group in algo["groups"].keys():
                    if group in group_list:
                        algos_conf[algo["name"]]["groups"][group] = algo[
                            "groups"
                        ][group]
                if "constraints" in algo:
                    algos_conf[algo["name"]]["constraints"] = algo[
                        "constraints"
                    ]

            if insert_algo:
                add_algo_group(named_groups)
            if insert_algo_group:
                add_algo_group(allowed_algo_groups[1])

    executables_to_run = dict()
    for algo in algos_conf.keys():
        validate_algorithm(algos_yaml, algo, gpu_present)
        for group in algos_conf[algo]["groups"].keys():
            executable = find_executable(
                algos_yaml, algo, group, k, batch_size
            )
            if executable not in executables_to_run:
                executables_to_run[executable] = {"index": []}
            build_params = algos_conf[algo]["groups"][group]["build"] or {}
            search_params = algos_conf[algo]["groups"][group]["search"] or {}

            param_names = []
            param_lists = []
            for param in build_params.keys():
                param_names.append(param)
                param_lists.append(build_params[param])

            all_build_params = itertools.product(*param_lists)

            search_param_names = []
            search_param_lists = []
            for search_param in search_params.keys():
                search_param_names.append(search_param)
                search_param_lists.append(search_params[search_param])

            for params in all_build_params:
                index = {"algo": algo, "build_param": {}}
                if group != "base":
                    index_name = f"{algo}_{group}"
                else:
                    index_name = f"{algo}"
                for i in range(len(params)):
                    index["build_param"][param_names[i]] = params[i]
                    index_name += "." + f"{param_names[i]}{params[i]}"

                if "constraints" in algos_conf[algo]:
                    if "build" in algos_conf[algo]["constraints"]:
                        importable = algos_conf[algo]["constraints"]["build"]
                        importable = importable.split(".")
                        module = ".".join(importable[:-1])
                        func = importable[-1]
                        validator = import_module(module)
                        build_constraints = getattr(validator, func)
                        if "dims" not in conf_file["dataset"]:
                            raise ValueError(
                                "`dims` needed for build constraints but not "
                                "specified in datasets.yaml"
                            )
                        if not build_constraints(
                            index["build_param"], conf_file["dataset"]["dims"]
                        ):
                            continue

                index["name"] = index_name
                index["file"] = os.path.join(
                    args.dataset_path, args.dataset, "index", index_name
                )
                index["search_params"] = []
                all_search_params = itertools.product(*search_param_lists)
                for search_params in all_search_params:
                    search_dict = dict()
                    for i in range(len(search_params)):
                        search_dict[search_param_names[i]] = search_params[i]
                    if "constraints" in algos_conf[algo]:
                        if "search" in algos_conf[algo]["constraints"]:
                            importable = algos_conf[algo]["constraints"][
                                "search"
                            ]
                            importable = importable.split(".")
                            module = ".".join(importable[:-1])
                            func = importable[-1]
                            validator = import_module(module)
                            search_constraints = getattr(validator, func)
                            if search_constraints(
                                search_dict,
                                index["build_param"],
                                k,
                                batch_size,
                            ):
                                index["search_params"].append(search_dict)
                    else:
                        index["search_params"].append(search_dict)
                executables_to_run[executable]["index"].append(index)

                if len(index["search_params"]) == 0:
                    print("No search parameters were added to configuration")

    run_build_and_search(
        conf_file,
        f"{args.dataset}",
        conf_filedir,
        executables_to_run,
        args.dataset_path,
        args.force,
        build,
        search,
        dry_run,
        k,
        batch_size,
        args.search_threads,
        mode,
        args.raft_log_level,
    )


if __name__ == "__main__":
    main()
