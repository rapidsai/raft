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


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", default="glove-100-inner")
    parser.add_argument("--output", help="Path to the PNG output file")
    parser.add_argument(
        "--x-scale",
        help="Scale to use when drawing the X-axis. \
              Typically linear, logit or a2", 
        default="linear"
    )
    parser.add_argument(
        "--y-scale",
        help="Scale to use when drawing the Y-axis",
        choices=["linear", "log", "symlog", "logit"],
        default="linear",
    )
    args = parser.parse_args()

    if not args.output:
        args.output = f"results/{args.dataset}.png"
        print(f"writing output to {args.output}")


if __name__ == "__main__":
    main()
