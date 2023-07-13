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


def export_results(output_filepath, recompute, groundtruth_filepath, result_filepaths):
    # result_filepaths = " ".join(result_filepaths)
    # print(result_filepaths)
    if recompute:
        p = subprocess.Popen(["scripts/eval.pl", "-f", "-o", output_filepath,
                              groundtruth_filepath] + result_filepaths)
    else:
        p = subprocess.Popen(["scripts/eval.pl", "-o", output_filepath,
                              groundtruth_filepath] + result_filepaths)
    p.wait()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", help="Path to the output file",
                        required=True)
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute metrics")
    parser.add_argument("--groundtruth",
                        help="Dataset whose groundtruth is used",
                        required=True)
    args, result_filepaths = parser.parse_known_args()
    
    # assume "result/<groundtruth_dataset>" folder to be default
    # if nothing is provided
    if len(result_filepaths) == 0:
        result_filepaths = ["result/%s" % args.groundtruth]

    groundtruth_filepath = os.path.join("data", args.groundtruth,
                                        "groundtruth.neighbors.ibin")
    export_results(args.output, args.recompute, groundtruth_filepath,
                   result_filepaths)


if __name__ == "__main__":
    main()