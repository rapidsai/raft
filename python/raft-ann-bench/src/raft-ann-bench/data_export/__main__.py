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

import pandas as pd


def read_file(dataset, dataset_path, method):
    dir = os.path.join(dataset_path, dataset, "result", method)
    for file in os.listdir(dir):
        if file.endswith(".json"):
            with open(os.path.join(dir, file), "r") as f:
                data = json.load(f)
                df = pd.DataFrame(data["benchmarks"])
                yield (os.path.join(dir, file), file.split("-")[0], df)


def convert_json_to_csv_build(dataset, dataset_path):
    for file, algo_name, df in read_file(dataset, dataset_path, "build"):
        df["name"] = df["name"].str.split("/").str[0]
        write = pd.DataFrame(
            {
                "algo_name": [algo_name] * len(df),
                "index_name": df["name"],
                "time": df["real_time"],
            }
        )
        write.to_csv(file.replace(".json", ".csv"), index=False)


def convert_json_to_csv_search(dataset, dataset_path):
    for file, algo_name, df in read_file(dataset, dataset_path, "search"):
        df["name"] = df["name"].str.split("/").str[0]
        write = pd.DataFrame(
            {
                "algo_name": [algo_name] * len(df),
                "index_name": df["name"],
                "recall": df["Recall"],
                "qps": df["items_per_second"],
            }
        )
        write.to_csv(file.replace(".json", ".csv"), index=False)


def main():

    call_path = os.getcwd()
    if "RAPIDS_DATASET_ROOT_DIR" in os.environ:
        default_dataset_path = os.getenv("RAPIDS_DATASET_ROOT_DIR")
    else:
        default_dataset_path = os.path.join(call_path, "datasets/")

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset", help="dataset to download", default="glove-100-inner"
    )
    parser.add_argument(
        "--dataset-path",
        help="path to dataset folder",
        default=default_dataset_path,
    )

    args = parser.parse_args()

    convert_json_to_csv_build(args.dataset, args.dataset_path)
    convert_json_to_csv_search(args.dataset, args.dataset_path)


if __name__ == "__main__":
    main()
