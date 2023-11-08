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
import sys
import warnings

import pandas as pd

skip_build_cols = set(
    [
        "algo_name",
        "index_name",
        "time",
        "name",
        "family_index",
        "per_family_instance_index",
        "run_name",
        "run_type",
        "repetitions",
        "repetition_index",
        "iterations",
        "real_time",
        "time_unit",
        "index_size",
    ]
)

skip_search_cols = (
    set(["recall", "qps", "items_per_second", "Recall"]) | skip_build_cols
)


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
        algo_name = algo_name.replace("_base", "")
        df["name"] = df["name"].str.split("/").str[0]
        write = pd.DataFrame(
            {
                "algo_name": [algo_name] * len(df),
                "index_name": df["name"],
                "time": df["real_time"],
            }
        )
        for name in df:
            if name not in skip_build_cols:
                write[name] = df[name]
        filepath = os.path.normpath(file).split(os.sep)
        filename = filepath[-1].split("-")[0] + ".csv"
        write.to_csv(
            os.path.join(f"{os.sep}".join(filepath[:-1]), filename),
            index=False,
        )


def convert_json_to_csv_search(dataset, dataset_path):
    for file, algo_name, df in read_file(dataset, dataset_path, "search"):
        build_file = os.path.join(
            dataset_path, dataset, "result", "build", f"{algo_name}.csv"
        )
        algo_name = algo_name.replace("_base", "")
        df["name"] = df["name"].str.split("/").str[0]
        write = pd.DataFrame(
            {
                "algo_name": [algo_name] * len(df),
                "index_name": df["name"],
                "recall": df["Recall"],
                "qps": df["items_per_second"],
            }
        )
        for name in df:
            if name not in skip_search_cols:
                write[name] = df[name]

        if os.path.exists(build_file):
            build_df = pd.read_csv(build_file)
            write_ncols = len(write.columns)
            write["build time"] = None
            write["build threads"] = None
            write["build cpu_time"] = None
            write["build GPU"] = None

            for col_idx in range(5, len(build_df.columns)):
                col_name = build_df.columns[col_idx]
                write[col_name] = None

            for s_index, search_row in write.iterrows():
                for b_index, build_row in build_df.iterrows():
                    if search_row["index_name"] == build_row["index_name"]:
                        write.iloc[s_index, write_ncols] = build_df.iloc[
                            b_index, 2
                        ]
                        write.iloc[s_index, write_ncols + 1 :] = build_df.iloc[
                            b_index, 3:
                        ]
                        break
        else:
            warnings.warn(
                f"Build CSV not found for {algo_name}, build params won't be "
                "appended in the Search CSV"
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

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    convert_json_to_csv_build(args.dataset, args.dataset_path)
    convert_json_to_csv_search(args.dataset, args.dataset_path)


if __name__ == "__main__":
    main()
