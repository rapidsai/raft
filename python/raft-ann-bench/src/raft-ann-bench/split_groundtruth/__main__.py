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


def split_groundtruth(groundtruth_filepath):
    ann_bench_scripts_path = "split_groundtruth.pl"
    pwd = os.getcwd()
    os.chdir("/".join(groundtruth_filepath.split("/")[:-1]))
    groundtruth_filename = groundtruth_filepath.split("/")[-1]
    subprocess.run(
        [ann_bench_scripts_path, groundtruth_filename, "groundtruth"],
        check=True,
    )
    os.chdir(pwd)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--groundtruth",
        help="Path to billion-scale dataset groundtruth file",
        required=True,
    )
    args = parser.parse_args()

    split_groundtruth(args.groundtruth)


if __name__ == "__main__":
    main()
