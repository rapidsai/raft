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
import os
import subprocess
import json

from pathlib import Path

def parse_filepaths(fs):
    for p in fs:
        if p.endswith(".json") and os.path.exists(p):
            yield p
        else:
            for f in Path(p).rglob('*.json'):
                yield f.as_posix()

def export_results(output_filepath, recompute, groundtruth_filepath,
                   result_filepath):
    print(f"Writing output file to: {output_filepath}")

    parsed_filepaths = parse_filepaths(result_filepaths)

    with open(output_filepath, 'w') as out:
        out.write("Algo,Recall,QPS\n")

        for fp in parsed_filepaths:
            with open(fp, 'r') as f:
                data = json.load(f)
                for benchmark_case in data["benchmarks"]:
                    algo = benchmark_case["name"]
                    recall = benchmark_case["Recall"]
                    qps = benchmark_case["items_per_second"]
                    out.write(f"{algo},{recall},{qps}\n")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", help="Path to the CSV output file",
                        required=True)
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute metrics")
    parser.add_argument("--dataset",
                        help="Name of the dataset to export results for",
                        default="glove-100-inner")
    parser.add_argument(
        "--dataset-path",
        help="path to dataset folder",
        default=os.path.join(os.getenv("RAFT_HOME"),
                             "bench", "ann", "data")
    )

    args, result_filepaths = parser.parse_known_args()

    # if nothing is provided
    if len(result_filepaths) == 0:
        raise ValueError("No filepaths to results were provided")

    groundtruth_filepath = os.path.join(args.dataset_path, args.dataset,
                                        "groundtruth.neighbors.ibin")
    export_results(args.output, args.recompute, groundtruth_filepath,
                   result_filepath)


if __name__ == "__main__":
    main()
