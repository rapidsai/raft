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
import yaml


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


def validate_algorithm(algos_conf, algo, gpu_present):
    algos_conf_keys = set(algos_conf.keys())
    if gpu_present:
        return algo in algos_conf_keys
    else:
        return (
            algo in algos_conf_keys
            and algos_conf[algo]["requires_gpu"] is False
        )


def find_executable(algos_conf, algo, k, batch_size):
    executable = algos_conf[algo]["executable"]

    build_path = os.getenv("RAFT_HOME")
    if build_path is not None:
        build_path = os.path.join(build_path, "cpp", "build", executable)
        if os.path.exists(build_path):
            print(f"-- Using RAFT bench from repository in {build_path}. ")
            return (executable, build_path, f"{algo}-{k}-{batch_size}")

    # if there is no build folder present, we look in the conda environment
    conda_path = os.getenv("CONDA_PREFIX")
    if conda_path is not None:
        conda_path = os.path.join(conda_path, "bin", "ann", executable)
        if os.path.exists(conda_path):
            print("-- Using RAFT bench found in conda environment. ")
            return (executable, conda_path, f"{algo}-{k}-{batch_size}")

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
    k,
    batch_size,
):
    for executable, ann_executable_path, algo in executables_to_run.keys():
        # Need to write temporary configuration
        temp_conf_filename = f"temporary_{conf_filename}"
        temp_conf_filepath = os.path.join(conf_filedir, temp_conf_filename)
        with open(temp_conf_filepath, "w") as f:
            temp_conf = dict()
            temp_conf["dataset"] = conf_file["dataset"]
            temp_conf["search_basic_param"] = conf_file["search_basic_param"]
            temp_conf["index"] = executables_to_run[
                (executable, ann_executable_path, algo)
            ]["index"]
            json.dump(temp_conf, f)

        legacy_result_folder = os.path.join(
            dataset_path, conf_file["dataset"]["name"], "result"
        )
        os.makedirs(legacy_result_folder, exist_ok=True)
        if build:
            build_folder = os.path.join(legacy_result_folder, "build")
            os.makedirs(build_folder, exist_ok=True)
            cmd = [
                ann_executable_path,
                "--build",
                "--data_prefix=" + dataset_path,
                "--benchmark_out_format=json",
                "--benchmark_out="
                + f"{os.path.join(build_folder, f'{algo}.json')}",
            ]
            if force:
                cmd = cmd + ["--overwrite"]
            cmd = cmd + [temp_conf_filepath]
            subprocess.run(cmd, check=True)

        if search:
            search_folder = os.path.join(legacy_result_folder, "search")
            os.makedirs(search_folder, exist_ok=True)
            cmd = [
                ann_executable_path,
                "--search",
                "--data_prefix=" + dataset_path,
                "--benchmark_counters_tabular",
                "--override_kv=k:%s" % k,
                "--override_kv=n_queries:%s" % batch_size,
                "--benchmark_min_warmup_time=0.01",
                "--benchmark_out_format=json",
                "--benchmark_out="
                + f"{os.path.join(search_folder, f'{algo}.json')}",
            ]
            if force:
                cmd = cmd + ["--overwrite"]
            cmd = cmd + [temp_conf_filepath]
            subprocess.run(cmd, check=True)

        # os.remove(temp_conf_filepath)


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
        help="path to configuration file for datasets",
    )
    parser.add_argument(
        "--configuration",
        help="path to configuration file or directory for algorithms",
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
                              algorithms",
        default=None,
    )
    # parser.add_argument(
    #     "--indices",
    #     help="run only comma separated list of named indices. \
    #                           parameter `algorithms` is ignored",
    #     default=None,
    # )
    parser.add_argument(
        "--groups",
        help="comma separated groups of parameters to run the benchmarks for",
        default="base"
    )
    parser.add_argument(
        "-f",
        "--force",
        help="re-run algorithms even if their results \
                              already exist",
        action="store_true",
    )

    args = parser.parse_args()

    # If both build and search are not provided,
    # run both
    if not args.build and not args.search:
        build = True
        search = True
    else:
        build = args.build
        search = args.search

    k = args.count
    batch_size = args.batch_size

    # Read configuration file associated to datasets
    if args.dataset_configuration:
        dataset_conf_f = args.dataset_configuration
    else:
        dataset_conf_f = os.path.join(
            scripts_path, "conf", "datasets.yaml"
        )
    with open(dataset_conf_f, "r") as f:
        dataset_conf_all = yaml.safe_load(f)

    for dataset in dataset_conf_all:
        if args.dataset == dataset["name"]:
            dataset_conf = dataset
    
    conf_file = dict()
    conf_file["dataset"] = dataset_conf
    if args.subset_size:
        conf_file["dataset"]["subset_size"] = args.subset_size

    conf_file["search_basic_param"] = {}
    conf_file["search_basic_param"]["k"] = k
    conf_file["search_basic_param"]["batch_size"] = batch_size

    algos_conf_fs = os.listdir(os.path.join(scripts_path, "conf", "algos"))
    algos_conf_fs = [os.path.join(scripts_path, "conf", "algos", f) \
                        for f in algos_conf_fs]
    conf_filedir = os.path.join(scripts_path, "conf", "algos")
    if args.configuration:
        if os.path.isdir(args.configuration):
            conf_filedir = args.configuration
            algos_conf_fs = algos_conf_fs + \
                            [os.path.join(args.configuration, f) \
                            for f in os.listdir(args.configuration)]
        elif os.path.isfile(args.configuration):
            conf_filedir = os.path.normpath(args.configuration).split(os.sep)
            algos_conf_fs = algos_conf_fs + [args.configuration]

    filter_algos = True if args.algorithms else False
    if filter_algos:
        allowed_algos = args.algorithms.split(",")
    named_groups = args.groups.split(",")
    algos_conf = dict()
    for algo_f in algos_conf_fs:
        with open(algo_f, "r") as f:
            if algo_f.split("/")[-1] == "raft_cagra.yaml":
                algo = yaml.safe_load(f)
                insert_algo = True
                if filter_algos:
                    if algo["name"] not in allowed_algos:
                        insert_algo = False
                if insert_algo:
                    if algo["name"] not in algos_conf:
                        algos_conf[algo["name"]] = dict()
                    for group in algo.keys():
                        if group != "name":
                            if group in named_groups:
                                algos_conf[algo["name"]][group] = algo[group]

    executables_to_run = dict()
    for algo in algos_conf.keys():
        validate_algorithm(algos_yaml, algo, gpu_present)
        executable = find_executable(algos_yaml, algo, k, batch_size)
        if executable not in executables_to_run:
            executables_to_run[executable] = {"index": []}
        for group in algos_conf[algo].keys():
            build_params = algos_conf[algo][group]["build"]
            search_params = algos_conf[algo][group]["search"]

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
                index_name = f"{algo}"
                for i in range(len(params)):
                    index["build_param"][param_names[i]] = params[i]
                    index_name += "." + f"{param_names[i]}{params[i]}"
                index["name"] = index_name
                index["file"] = os.path.join(args.dataset_path, args.dataset, 
                                             "index", index_name)
                index["search_params"] = []
                all_search_params = itertools.product(*search_param_lists)
                for search_params in all_search_params:
                    search_dict = dict()
                    for i in range(len(search_params)):
                        search_dict[search_param_names[i]] = search_params[i]
                    index["search_params"].append(search_dict)
                executables_to_run[executable]["index"].append(index)

    print(executables_to_run)
    # conf_filename = conf_filepath.split("/")[-1]
    # conf_filedir = "/".join(conf_filepath.split("/")[:-1])
    # dataset_path = args.dataset_path
    # if not os.path.exists(conf_filepath):
    #     raise FileNotFoundError(conf_filename)

    # with open(conf_filepath, "r") as f:
    #     conf_file = json.load(f)

    # dataset_name = conf_file["dataset"]["name"]

    # executables_to_run = dict()
    # # At least one named index should exist in config file
    # if args.indices:
    #     indices = set(args.indices.split(","))
    #     # algo associated with index should still be present in algos.yaml
    #     # and enabled
    #     for index in conf_file["index"]:
    #         curr_algo = index["algo"]
    #         if index["name"] in indices and validate_algorithm(
    #             algos_conf, curr_algo, gpu_present
    #         ):
    #             executable_path = find_executable(
    #                 algos_conf, curr_algo, k, batch_size
    #             )
    #             if executable_path not in executables_to_run:
    #                 executables_to_run[executable_path] = {"index": []}
    #             executables_to_run[executable_path]["index"].append(index)

    # # switch to named algorithms if indices parameter is not supplied
    # elif args.algorithms:
    #     algorithms = set(args.algorithms.split(","))
    #     # pick out algorithms from conf file that exist
    #     # and are enabled in algos.yaml
    #     for index in conf_file["index"]:
    #         curr_algo = index["algo"]
    #         if curr_algo in algorithms and validate_algorithm(
    #             algos_conf, curr_algo, gpu_present
    #         ):
    #             executable_path = find_executable(
    #                 algos_conf, curr_algo, k, batch_size
    #             )
    #             if executable_path not in executables_to_run:
    #                 executables_to_run[executable_path] = {"index": []}
    #             executables_to_run[executable_path]["index"].append(index)

    # # default, try to run all available algorithms
    # else:
    #     for index in conf_file["index"]:
    #         curr_algo = index["algo"]
    #         if validate_algorithm(algos_conf, curr_algo, gpu_present):
    #             executable_path = find_executable(
    #                 algos_conf, curr_algo, k, batch_size
    #             )
    #             if executable_path not in executables_to_run:
    #                 executables_to_run[executable_path] = {"index": []}
    #             executables_to_run[executable_path]["index"].append(index)

    # # Replace index to dataset path
    # for executable_path in executables_to_run:
    #     for pos, index in enumerate(
    #         executables_to_run[executable_path]["index"]
    #     ):
    #         index["file"] = os.path.join(
    #             dataset_path, dataset_name, "index", index["name"]
    #         )
    #         executables_to_run[executable_path]["index"][pos] = index

    print(conf_filedir)
    run_build_and_search(
        conf_file,
        f"{args.dataset}.json",
        conf_filedir,
        executables_to_run,
        args.dataset_path,
        args.force,
        build,
        search,
        k,
        batch_size,
    )


if __name__ == "__main__":
    main()
