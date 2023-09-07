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
from urllib.request import urlretrieve


def get_dataset_path(name, ann_bench_data_path):
    if not os.path.exists(ann_bench_data_path):
        os.mkdir(ann_bench_data_path)
    return os.path.join(ann_bench_data_path, f"{name}.hdf5")


def download_dataset(url, path):
    if not os.path.exists(path):
        print(f"downloading {url} -> {path}...")
        urlretrieve(url, path)


def convert_hdf5_to_fbin(path, normalize):
    scripts_path = os.path.dirname(os.path.realpath(__file__))
    ann_bench_scripts_path = os.path.join(scripts_path, "hdf5_to_fbin.py")
    print(f"calling script {ann_bench_scripts_path}")
    if normalize and "angular" in path:
        p = subprocess.Popen(
            ["python", ann_bench_scripts_path, "-n", "%s" % path]
        )
    else:
        p = subprocess.Popen(["python", ann_bench_scripts_path, "%s" % path])
    p.wait()


def move(name, ann_bench_data_path):
    if "angular" in name:
        new_name = name.replace("angular", "inner")
    else:
        new_name = name
    new_path = os.path.join(ann_bench_data_path, new_name)
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    for bin_name in [
        "base.fbin",
        "query.fbin",
        "groundtruth.neighbors.ibin",
        "groundtruth.distances.fbin",
    ]:
        os.rename(
            f"{ann_bench_data_path}/{name}.{bin_name}",
            f"{new_path}/{bin_name}",
        )


def download(name, normalize, ann_bench_data_path):
    path = get_dataset_path(name, ann_bench_data_path)
    try:
        url = f"http://ann-benchmarks.com/{name}.hdf5"
        download_dataset(url, path)

        convert_hdf5_to_fbin(path, normalize)

        move(name, ann_bench_data_path)
    except Exception:
        print(f"Cannot download {url}")
        raise


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
        "--dataset", help="dataset to download", default="glove-100-angular"
    )
    parser.add_argument(
        "--dataset-path",
        help="path to download dataset",
        default=default_dataset_path,
    )
    parser.add_argument(
        "--normalize",
        help="normalize cosine distance to inner product",
        action="store_true",
    )
    args = parser.parse_args()

    download(args.dataset, args.normalize, args.dataset_path)


if __name__ == "__main__":
    main()
