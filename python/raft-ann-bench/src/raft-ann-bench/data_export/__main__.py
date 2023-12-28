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
import traceback
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
    set(["recall", "qps", "latency", "items_per_second", "Recall", "Latency"])
    | skip_build_cols
)

metrics = {
    "k-nn": {
        "description": "Recall",
        "worst": float("-inf"),
        "lim": [0.0, 1.03],
    },
    "throughput": {
        "description": "Queries per second (1/s)",
        "worst": float("-inf"),
    },
    "latency": {
        "description": "Search Latency (s)",
        "worst": float("inf"),
    },
}


def read_file(dataset, dataset_path, method):
    dir = os.path.join(dataset_path, dataset, "result", method)
    for file in os.listdir(dir):
        if file.endswith(".json"):
            with open(
                os.path.join(dir, file), "r", encoding="ISO-8859-1"
            ) as f:
                try:
                    data = json.load(f)
                    df = pd.DataFrame(data["benchmarks"])
                    filename_split = file.split(",")
                    algo_name = (filename_split[0], filename_split[1])
                    yield os.path.join(dir, file), algo_name, df
                except Exception as e:
                    print(
                        "An error occurred processing file %s (%s). "
                        "Skipping..." % (file, e)
                    )


def convert_json_to_csv_build(dataset, dataset_path):
    for file, algo_name, df in read_file(dataset, dataset_path, "build"):
        try:
            if "base" in algo_name[1]:
                algo_name = algo_name[0]
            else:
                algo_name = "_".join(algo_name)
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
            write.to_csv(file.replace(".json", ".csv"), index=False)
        except Exception as e:
            print(
                "An error occurred processing file %s (%s). Skipping..."
                % (file, e)
            )
            traceback.print_exc()


def create_pointset(data, xn, yn):
    xm, ym = (metrics[xn], metrics[yn])
    rev_y = -1 if ym["worst"] < 0 else 1
    rev_x = -1 if xm["worst"] < 0 else 1

    y_idx = 3 if yn == "throughput" else 4
    data.sort(key=lambda t: (rev_y * t[y_idx], rev_x * t[2]))

    lines = []
    last_x = xm["worst"]
    comparator = (
        (lambda xv, lx: xv > lx) if last_x < 0 else (lambda xv, lx: xv < lx)
    )
    for d in data:
        if comparator(d[2], last_x):
            last_x = d[2]
            lines.append(d)
    return lines


def get_frontier(df, metric):
    lines = create_pointset(df.values.tolist(), "k-nn", metric)
    return pd.DataFrame(lines, columns=df.columns)


def convert_json_to_csv_search(dataset, dataset_path):
    for file, algo_name, df in read_file(dataset, dataset_path, "search"):
        try:
            build_file = os.path.join(
                dataset_path,
                dataset,
                "result",
                "build",
                f"{','.join(algo_name)}.csv",
            )
            print(build_file)
            if "base" in algo_name[1]:
                algo_name = algo_name[0]
            else:
                algo_name = "_".join(algo_name)
            df["name"] = df["name"].str.split("/").str[0]
            try:
                write = pd.DataFrame(
                    {
                        "algo_name": [algo_name] * len(df),
                        "index_name": df["name"],
                        "recall": df["Recall"],
                        "throughput": df["items_per_second"],
                        "latency": df["Latency"],
                    }
                )
            except Exception as e:
                print(
                    "Search file %s (%s) missing a key. Skipping..."
                    % (file, e)
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

                try:
                    for col_idx in range(6, len(build_df.columns)):
                        col_name = build_df.columns[col_idx]
                        write[col_name] = None

                    for s_index, search_row in write.iterrows():
                        for b_index, build_row in build_df.iterrows():
                            if (
                                search_row["index_name"]
                                == build_row["index_name"]
                            ):
                                write.iloc[
                                    s_index, write_ncols
                                ] = build_df.iloc[b_index, 2]
                                write.iloc[
                                    s_index, write_ncols + 1 :
                                ] = build_df.iloc[b_index, 3:]
                                break
                except Exception as e:
                    print(
                        "Build file %s (%s) missing a key. Skipping..."
                        % (build_file, e)
                    )
            else:
                warnings.warn(
                    f"Build CSV not found for {algo_name}, "
                    f"build params won't be "
                    "appended in the Search CSV"
                )

            write.to_csv(file.replace(".json", ",raw.csv"), index=False)
            throughput = get_frontier(write, "throughput")
            throughput.to_csv(
                file.replace(".json", ",throughput.csv"), index=False
            )
            latency = get_frontier(write, "latency")
            latency.to_csv(file.replace(".json", ",latency.csv"), index=False)
        except Exception as e:
            print(
                "An error occurred processing file %s (%s). Skipping..."
                % (file, e)
            )
            traceback.print_exc()


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
