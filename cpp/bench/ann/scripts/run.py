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

def main():
    # Read list of allowed algorithms
    with open("algos.yaml", "r") as f:
        try:
            algos_conf = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)


    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--dataset",
        help="the dataset to load training points from",
        default="glove-100-inner",
    )
    parser.add_argument("--algorithms", help="run only comma seperated list of named algorithms", default=None)
    parser.add_argument("--indices", help="run only comma separated list of named indices. parameter `algorithms` is ignored", default=None)
    parser.add_argument("--force", help="re-run algorithms even if their results already exist", action="store_true")

    args = parser.parse_args()

    # Read configuration file associated to dataset
    conf_filename = os.path.join("conf", "%s.json" % args.dataset)
    if not os.path.exists(conf_filename):
        raise FileNotFoundError(conf_filename)
    
    with open(conf_filename, "r") as f:
        conf_file = json.load(f)

    # Ensure base and query files exist for dataset
    if not os.path.exists(conf_file["dataset"]["base_file"]):
        raise FileNotFoundError(conf_file["dataset"]["base_file"])
    if not os.path.exists(conf_file["dataset"]["query_file"]):
        raise FileNotFoundError(conf_file["dataset"]["query_file"])

    # At least one named index should exist in config file
    if args.indices:
        indices = set(args.indices.split(","))
        temporary_conf = conf_file.copy()
        found_pos = []
        for pos, index in enumerate(temporary_conf["index"]):
            if index["name"] in indices:
                found_pos.append(pos)

        # filter available indices
        if len(found_pos) == 0:
            raise Exception("No named indices found in %s" % conf_filename)
        temporary_conf["index"] = [temporary_conf["index"][p] for p in found_pos]
    # switch to named algorithms if indices parameter is not supplied
    elif args.algorithms:
        algorithms = set(args.algorithms.split(","))
        # pick out algorithms from conf file that exist
        # and are enabled in algos.yaml
        temporary_conf = conf_file.copy()
        algos_conf_keys = set(algos_conf.keys())
        found_pos = []
        for pos, index in enumerate(conf_file["index"]):
            curr_algo = index["algo"]
            if curr_algo in algorithms or (curr_algo in algos_conf_keys and algos_conf[curr_algo]["disabled"]):
                found_pos.append(pos)
        # filter available algorithms
        if len(found_pos) == 0:
            raise Exception("No named algorithms found in %s" % conf_filename)
        temporary_conf["index"] = [temporary_conf["index"][p] for p in found_pos]

if __name__ == "__main__":
    main()