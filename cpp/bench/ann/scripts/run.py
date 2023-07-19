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


def validate_algorithm_and_executable(algos_conf, algo):
    algos_conf_keys = set(algos_conf.keys())
    if algo in algos_conf_keys and not algos_conf[algo]["disabled"]:
        # executable is assumed to be in folder "<root>/cpp/build"
        executable_filepath = f"../../build/{algos_conf[algo]['executable']}"
        if not os.path.exists(executable_filepath):
            raise FileNotFoundError(executable_filepath)
        return True
    else:
        return False


def run_build_and_search(conf_filename, conf_file, executables_to_run, force):
    # Need to write temporary configuration
    temp_conf_filename = f"temporary_{conf_filename}"
    temp_conf_filepath = os.path.join("conf", temp_conf_filename)
    with open(temp_conf_filepath, "w") as f:
        json.dump(conf_file, f)

    print(f"Building indices for configuration {conf_filename}")
    for executable in executables_to_run:
        if force:
            p = subprocess.Popen([f"../../build/{executable}", "-b", "-f",
                                  temp_conf_filepath])
            p.wait()
        else:
            p = subprocess.Popen([f"../../build/{executable}", "-b",
                                  temp_conf_filepath])
            p.wait()

    print(f"Searching indices for configuration {conf_filename}")
    for executable in executables_to_run:
        if force:
            p = subprocess.Popen([f"../../build/{executable}", "-s", "-f",
                                  temp_conf_filepath])
            p.wait()
        else:
            p = subprocess.Popen([f"../../build/{executable}", "-s",
                                  temp_conf_filepath])
            p.wait()

    os.remove(temp_conf_filepath)


def main():
    # Read list of allowed algorithms
    with open("algos.yaml", "r") as f:
        algos_conf = yaml.safe_load(f)

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--dataset",
        help="the dataset to load training points from",
        default="glove-100-inner",
    )
    parser.add_argument("--algorithms",
                        help="run only comma separated list of named \
                              algorithms",
                        default=None)
    parser.add_argument("--indices",
                        help="run only comma separated list of named indices. \
                              parameter `algorithms` is ignored",
                        default=None)
    parser.add_argument("--force",
                        help="re-run algorithms even if their results \
                              already exist",
                        action="store_true")

    args = parser.parse_args()

    # Read configuration file associated to dataset
    conf_filename = f"{args.dataset}.json"
    conf_filepath = os.path.join("conf", conf_filename)
    if not os.path.exists(conf_filepath):
        raise FileNotFoundError(conf_filename)

    with open(conf_filepath, "r") as f:
        conf_file = json.load(f)

    # Ensure base and query files exist for dataset
    if not os.path.exists(conf_file["dataset"]["base_file"]):
        raise FileNotFoundError(conf_file["dataset"]["base_file"])
    if not os.path.exists(conf_file["dataset"]["query_file"]):
        raise FileNotFoundError(conf_file["dataset"]["query_file"])

    temporary_conf = conf_file.copy()
    found_pos = []
    executables_to_run = set()
    # At least one named index should exist in config file
    if args.indices:
        indices = set(args.indices.split(","))
        # algo associated with index should still be present in algos.yaml
        # and enabled
        for pos, index in enumerate(conf_file["index"]):
            curr_algo = index["algo"]
            if index["name"] in indices and \
                    validate_algorithm_and_executable(algos_conf, curr_algo):
                found_pos.append(pos)
                executables_to_run.add(algos_conf[curr_algo]["executable"])

    # switch to named algorithms if indices parameter is not supplied
    elif args.algorithms:
        algorithms = set(args.algorithms.split(","))
        # pick out algorithms from conf file that exist
        # and are enabled in algos.yaml
        for pos, index in enumerate(conf_file["index"]):
            curr_algo = index["algo"]
            if curr_algo in algorithms and \
                    validate_algorithm_and_executable(algos_conf, curr_algo):
                found_pos.append(pos)
                executables_to_run.add(algos_conf[curr_algo]["executable"])

    # default, try to run all available algorithms
    else:
        for pos, index in enumerate(conf_file["index"]):
            curr_algo = index["algo"]
            if validate_algorithm_and_executable(algos_conf, curr_algo):
                found_pos.append(pos)
                executables_to_run.add(algos_conf[curr_algo]["executable"])

    # filter available algorithms or indices
    if len(found_pos) == 0:
        raise Exception(f"No named indices/algorithms found in {conf_filename}"
                        )
    temporary_conf["index"] = [temporary_conf["index"][p] for p in found_pos]

    run_build_and_search(conf_filename, temporary_conf, executables_to_run,
                         args.force)


if __name__ == "__main__":
    main()
