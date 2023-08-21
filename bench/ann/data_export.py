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


def export_results(output_filepath, recompute, groundtruth_filepath,
                   result_filepaths):
    print(f"Writing output file to: {output_filepath}")
    ann_bench_scripts_dir = os.path.join(os.getenv("RAFT_HOME"),
                                         "cpp/bench/ann/scripts")
    ann_bench_scripts_path = os.path.join(ann_bench_scripts_dir,
                                          "eval.pl")
    if recompute:
        p = subprocess.Popen([ann_bench_scripts_path, "-f", "-o", output_filepath,
                              groundtruth_filepath] + result_filepaths)
    else:
        p = subprocess.Popen([ann_bench_scripts_path, "-o", output_filepath,
                              groundtruth_filepath] + result_filepaths)
    p.wait()


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
    
    args = parser.parse_known_args()
    result_filepath = os.path.join(args.dataset_path, "result")

    groundtruth_filepath = os.path.join(args.dataset_path, args.dataset, 
                                        "groundtruth.neighbors.ibin")
    export_results(args.output, args.recompute, groundtruth_filepath,
                   result_filepath)


if __name__ == "__main__":
    main()
