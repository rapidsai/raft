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
import json
import os
import subprocess
import yaml


def validate_algorithm(algos_conf, algo):
    algos_conf_keys = set(algos_conf.keys())
    return algo in algos_conf_keys and not algos_conf[algo]["disabled"]


def find_executable():
    executable = "ANN_BENCH"
    conda_path = os.path.join(os.getenv("CONDA_PREFIX"), "bin", "ann",
                              executable)
    build_path = os.path.join(os.getenv("RAFT_HOME"), "cpp", "build", executable)
    if os.path.exists(conda_path):
        return conda_path
    elif os.path.exists(build_path):
        return build_path
    else:
        raise FileNotFoundError(executable)


def run_build_and_search(conf_filename, conf_file, dataset_path,
                         force, conf_filedir, build, search):
    ann_executable_path = find_executable()

    # Need to write temporary configuration
    temp_conf_filename = f"temporary_{conf_filename}"
    temp_conf_filepath = os.path.join(conf_filedir, temp_conf_filename)
    with open(temp_conf_filepath, "w") as f:
        json.dump(conf_file, f)

    data_prefix = "/".join(dataset_path.split("/")[:-1])
    if build:
        cmd = [ann_executable_path, "--build", "--data_prefix="+data_prefix]
        if force:
            cmd = cmd + ["--overwrite"]
        cmd = cmd + [temp_conf_filepath]
        print(cmd)
        p = subprocess.Popen(cmd)
        p.wait()

    if search:
        cmd = [ann_executable_path, "--search", "--benchmark_out_format=csv",
               "--benchmark_out=" + os.path.join(dataset_path, "result.csv"),
               "--data_prefix=" + data_prefix]
        if force:
            cmd = cmd + ["--overwrite"]
        cmd = cmd + [temp_conf_filepath]
        p = subprocess.Popen(cmd)
        p.wait()

    os.remove(temp_conf_filepath)


def main():
    scripts_path = os.path.dirname(os.path.realpath(__file__))
    # Read list of allowed algorithms
    with open(f"{scripts_path}/algos.yaml", "r") as f:
        algos_conf = yaml.safe_load(f)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--configuration",
        help="path to configuration file for a dataset",
    )
    parser.add_argument(
        "--dataset",
        help="dataset whose configuration file will be used",
    )
    parser.add_argument(
        "--dataset-path",
        help="path to dataset folder",
        default=os.path.join(os.getenv("RAFT_HOME"), 
                             "bench", "ann", "data")
    )
    parser.add_argument(
        "--build",
        action="store_true"
    )
    parser.add_argument(
        "--search",
        action="store_true"
    )
    parser.add_argument("--algorithms",
                        help="run only comma separated list of named \
                              algorithms",
                        default=None)
    parser.add_argument("--indices",
                        help="run only comma separated list of named indices. \
                              parameter `algorithms` is ignored",
                        default=None)
    parser.add_argument("-f", "--force",
                        help="re-run algorithms even if their results \
                              already exist",
                        action="store_true")
    parser.add_argument("--batch-size",
                        help="batch size for querying",
                        default=1)
    parser.add_argument("--k",
                        help="k neighbors",
                        default=10)

    args = parser.parse_args()

    # If both build and search are not provided,
    # run both
    if not args.build and not args.search:
        build = True
        search = True
    else:
        build = args.build
        search = args.search

    # Read configuration file associated to dataset
    if args.configuration:
        conf_filepath = args.configuration
    elif args.dataset:
        conf_filepath = \
            os.path.join(scripts_path, "conf", f"{args.dataset}.json")
    else:
        raise ValueError("One of parameters `configuration` or \
                         `dataset` need to be provided")
    conf_filename = conf_filepath.split("/")[-1]
    conf_filedir = "/".join(conf_filepath.split("/")[:-1])
    dataset = conf_filename.replace(".json", "")
    dataset_path = os.path.join(args.dataset_path, dataset)
    if not os.path.exists(conf_filepath):
        raise FileNotFoundError(conf_filename)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(dataset_path)

    with open(conf_filepath, "r") as f:
        conf_file = json.load(f)

    # # Replace base, query, gr to dataset-path
    # conf_file["dataset"]["base_file"] = os.path.join(dataset_path, "base.fbin")
    # conf_file["dataset"]["query_file"] = os.path.join(dataset_path, "query.fbin")
    # conf_file["dataset"]["groundtruth_neighbors_file"] = \
    #     os.path.join(dataset_path, "groundtruth.neighbors.ibin")
    # # Ensure base and query files exist for dataset
    # if not os.path.exists(conf_file["dataset"]["base_file"]):
    #     raise FileNotFoundError(conf_file["dataset"]["base_file"])
    # if not os.path.exists(conf_file["dataset"]["query_file"]):
    #     raise FileNotFoundError(conf_file["dataset"]["query_file"])
    # if not os.path.exists(conf_file["dataset"]["groundtruth_neighbors_file"]):
    #     raise FileNotFoundError(conf_file["dataset"]["groundtruth_neighbors_file"])

    # executables_to_run = dict()
    indices_to_run = []
    # At least one named index should exist in config file
    if args.indices:
        indices = set(args.indices.split(","))
        # algo associated with index should still be present in algos.yaml
        # and enabled
        for pos, index in enumerate(conf_file["index"]):
            curr_algo = index["algo"]
            if index["name"] in indices and \
                    validate_algorithm(algos_conf, curr_algo):
                # executable_path = find_executable(algos_conf, curr_algo)
                # if executable_path not in executables_to_run:
                #     executables_to_run[executable_path] = {"index": []}
                # executables_to_run[executable_path]["index"].append(index)
                indices_to_run.append(pos)

    # switch to named algorithms if indices parameter is not supplied
    elif args.algorithms:
        algorithms = set(args.algorithms.split(","))
        # pick out algorithms from conf file that exist
        # and are enabled in algos.yaml
        for pos, index in enumerate(conf_file["index"]):
            curr_algo = index["algo"]
            if curr_algo in algorithms and \
                    validate_algorithm(algos_conf, curr_algo):
                # executable_path = find_executable(algos_conf, curr_algo)
                # if executable_path not in executables_to_run:
                #     executables_to_run[executable_path] = {"index": []}
                # executables_to_run[executable_path]["index"].append(index)
                indices_to_run.append(pos)

    # default, try to run all available algorithms
    else:
        for pos, index in enumerate(conf_file["index"]):
            curr_algo = index["algo"]
            if validate_algorithm(algos_conf, curr_algo):
                # executable_path = find_executable(algos_conf, curr_algo)
                # if executable_path not in executables_to_run:
                #     executables_to_run[executable_path] = {"index": []}
                # executables_to_run[executable_path]["index"].append(index)
                indices_to_run.append(pos)

    # filter available indices
    if len(indices_to_run) == 0:
        raise ValueError("No indices found to run")
    conf_file["index"] = [conf_file["index"][i] for i in indices_to_run]

    # Replace index build to dataset path
    for pos, index in enumerate(conf_file["index"]):
        index["file"] = os.path.join(dataset_path, "index", index["name"])
        conf_file["index"][pos] = index

    print(conf_file)

    run_build_and_search(conf_filename, conf_file, dataset_path,
                         args.force, conf_filedir, build, search)


if __name__ == "__main__":
    main()
