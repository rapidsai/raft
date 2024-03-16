# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#
from __future__ import annotations

import dataclasses
import json
import logging
import math
import pathlib
import shlex
import shutil
import subprocess
import tempfile
import unittest

import cupy as cp
import numpy as np
import sklearn.datasets
import yaml
from raft_ann_bench import common

LOGGER = logging.getLogger(__name__)

TEST_DATA_PREFIX = pathlib.Path(__file__).parent.parent / "data"


def generate_dataset(
    filename, n_samples, n_features, dtype, rng
) -> pathlib.Path:
    """Reference: https://github.com/rapidsai/raft/pull/2023/"""
    memmap_bin_file = common.import_callable(
        "raft-ann-bench.generate_groundtruth.utils.memmap_bin_file"
    )

    fp = memmap_bin_file(
        filename, dtype, shape=(n_samples, n_features), mode="w+"
    )
    dtype = fp.dtype
    itemsize = fp.dtype.itemsize
    total_size = n_samples * n_features * itemsize / (1 << 30)

    print(
        "Generating dataset {0} shape=({1},{2}), dtype={3}, size={4:6.1f} "
        "GiB".format(filename, n_samples, n_features, dtype, total_size)
    )

    n = 1000000
    i = 0
    while i < n_samples:
        n_batch = n if i + n <= n_samples else n_samples - i
        if rng == "blobs":
            tmp, _ = sklearn.datasets.make_blobs(
                n_samples=n_batch,
                n_features=n_features,
                centers=int(math.sqrt(n_samples)),
                cluster_std=3,
                shuffle=True,
                random_state=1234,
            )
            tmp = tmp.astype(dtype)
        else:
            tmp = cp.random.uniform(size=(n_batch, n_features)).astype(dtype)
        fp[i : i + n_batch, :] = cp.asnumpy(tmp)
        i += n_batch
        print(
            "Step {0}/{1}: {2:6.1f} GiB written".format(
                i // n, n_samples // n, i * n_features * itemsize / (1 << 30)
            )
        )

    fp.flush()
    del fp
    return pathlib.Path(filename).resolve()


def generate_groundtruth(
    dataset_file: str | pathlib.Path,
    n_samples,
    n_features,
    dtype,
    n_queries,
    k,
    metric="euclidean",
) -> dict[str, pathlib.Path]:
    memmap_bin_file = common.import_callable(
        "raft-ann-bench.generate_groundtruth.utils.memmap_bin_file"
    )
    suffix_from_dtype = common.import_callable(
        "raft-ann-bench.generate_groundtruth.utils.suffix_from_dtype"
    )
    write_bin = common.import_callable(
        "raft-ann-bench.generate_groundtruth.utils.write_bin"
    )
    calc_truth = common.import_callable(
        "raft-ann-bench.generate_groundtruth.__main__.calc_truth"
    )
    generate_random_queries = common.import_callable(
        "raft-ann-bench.generate_groundtruth.__main__.generate_random_queries"
    )

    dataset_file = pathlib.Path(dataset_file)
    output_dir = dataset_file.parent
    query_file = output_dir / f"queries{suffix_from_dtype(dtype)}"
    groundtruth_neighbors_file = output_dir / "groundtruth.neighbors.ibin"
    groundtruth_distances_file = output_dir / "groundtruth.distances.fbin"

    dataset = memmap_bin_file(
        dataset_file, dtype, shape=(n_samples, n_features)
    )
    queries = generate_random_queries(n_queries, n_features, dtype)
    distances, indices = calc_truth(dataset, queries, k, metric)
    write_bin(query_file, queries)
    write_bin(groundtruth_neighbors_file, indices.astype(np.uint32))
    write_bin(groundtruth_distances_file, distances.astype(np.float32))
    # cmd = f"""python -m raft-ann-bench.generate_groundtruth \
    #     {dataset_file} \
    #     --output={output_dir} \
    #     --queries=random \
    #     --n_queries={n_queries}"""
    # subprocess.run(shlex.split(cmd), check=True)

    return dict(
        base=dataset_file,
        query=query_file,
        groundtruth_neighbors=groundtruth_neighbors_file,
    )


def run_build_old(
    data_prefix: str,
    dataset: str,
    algos: str,
    dataset_config_file: str | None = None,
    algo_config_file: str | None = None,
):
    cmd = f"""python -m raft-ann-bench.run \
        --dataset {dataset} --algorithms {algos} \
        -k 100 --batch-size 1\
        --build --force"""
    if dataset_config_file:
        cmd += f" --dataset-configuration {dataset_config_file}"
    if algo_config_file:
        cmd += f" --configuration {algo_config_file}"

    LOGGER.info(cmd)
    subprocess.run(
        shlex.split(cmd),
        check=True,
        cwd=pathlib.Path(data_prefix).resolve().parent,
    )


def run_build_new(
    data_prefix: str,
    dataset: str,
    algos: str,
    dataset_config_file: str | None = None,
    algo_config_file: str | None = None,
):
    run_part = f"""raft-ann-bench run \
        --dataset {dataset} --algorithms {algos} \
        --k 100 --batch-size 1"""
    if dataset_config_file:
        run_part += f" --dataset-config {dataset_config_file}"
    if algo_config_file:
        run_part += f" --algorithm-config {algo_config_file}"

    cmd = f"""{run_part} build --force"""
    LOGGER.info(cmd)
    subprocess.run(
        shlex.split(cmd),
        check=True,
        cwd=pathlib.Path(data_prefix).resolve().parent,
    )


def run_search_old(
    data_prefix: str,
    dataset: str,
    algos: str,
    dataset_config_file: str | None = None,
    algo_config_file: str | None = None,
):
    cmd = f"""python -m raft-ann-bench.run \
        --dataset {dataset} --algorithms {algos} \
        -k 100 --batch-size 1 \
        --search --search-mode latency --search-threads 1"""
    if dataset_config_file:
        cmd += f" --dataset-configuration {dataset_config_file}"
    if algo_config_file:
        cmd += f" --configuration {algo_config_file}"

    LOGGER.info(cmd)
    subprocess.run(
        shlex.split(cmd),
        check=True,
        cwd=pathlib.Path(data_prefix).resolve().parent,
    )


def run_search_new(
    data_prefix: str,
    dataset: str,
    algos: str,
    dataset_config_file: str | None = None,
    algo_config_file: str | None = None,
):
    run_part = f"""raft-ann-bench run \
        --dataset {dataset} --algorithms {algos} \
        --k 100 --batch-size 1"""
    if dataset_config_file:
        run_part += f" --dataset-config {dataset_config_file}"
    if algo_config_file:
        run_part += f" --algorithm-config {algo_config_file}"

    cmd = f"""{run_part} search --mode latency --threads 1"""
    LOGGER.info(cmd)
    subprocess.run(
        shlex.split(cmd),
        check=True,
        cwd=pathlib.Path(data_prefix).resolve().parent,
    )


class TestRunMigration(unittest.TestCase):
    @dataclasses.dataclass
    class DataPaths:
        base: pathlib.Path
        query: pathlib.Path
        groundtruth_neighbors: pathlib.Path
        dataset_configs: pathlib.Path

    def setUp(self):
        self.temp_dir = pathlib.Path(tempfile.mkdtemp())
        LOGGER.info(f"Temporary directory: {self.temp_dir}")

        self.dataset = "test-dataset"
        self.algo = "faiss_cpu_ivf_flat"
        self.k = 100
        # We will reuse the same generated data
        self.data_prefix_common = self.temp_dir / "common" / "datasets"

    def tearDown(self):
        # shutil.rmtree(self.temp_dir)
        pass

    def generate_data(self) -> DataPaths:
        n_samples = 100000
        n_queries = 50000
        n_features = 128
        dtype = cp.float32
        dataset_file = generate_dataset(
            filename=self.data_prefix_common / self.dataset / "base.fbin",
            n_samples=n_samples,
            n_features=n_features,
            dtype=dtype,
            rng="blobs",
        )
        data_paths = generate_groundtruth(
            dataset_file,
            n_samples=n_samples,
            n_features=n_features,
            n_queries=n_queries,
            k=self.k,
            dtype=dtype,
        )
        dataset_configs = [
            dict(
                name=self.dataset,
                base_file="{}/{}".format(
                    self.dataset, data_paths["base"].name
                ),
                subset_size=n_queries,
                dims=n_features,
                query_file="{}/{}".format(
                    self.dataset, data_paths["query"].name
                ),
                groundtruth_neighbors_file="{}/{}".format(
                    self.dataset, data_paths["groundtruth_neighbors"].name
                ),
                distance="euclidean",
            )
        ]
        dataset_configs_path = dataset_file.parent / "datasets.yaml"
        dataset_configs_path.write_text(yaml.dump(dataset_configs))
        return self.DataPaths(
            **data_paths, dataset_configs=dataset_configs_path
        )

    def compare_build_output(
        self, data_prefix_old: pathlib.Path, data_prefix_new: pathlib.Path
    ):
        # Compare the build outputs
        rel_build_path = (
            pathlib.Path(self.dataset)
            / "result"
            / "build"
            / f"{self.algo},base.json"
        )

        old_build_file = data_prefix_old / rel_build_path
        new_build_file = data_prefix_new / rel_build_path
        self.assertTrue(old_build_file.exists())
        self.assertTrue(new_build_file.exists())

        old_build = json.loads(old_build_file.read_text())
        new_build = json.loads(new_build_file.read_text())

        # Context should be almost the same
        old_context = {**old_build["context"]}
        new_context = {**new_build["context"]}
        for key in ("date", "load_avg"):
            for ctx in [old_context, new_context]:
                del ctx[key]
        self.assertDictEqual(old_context, new_context)

        # Benchmarks should be almost the same
        old_benchmarks = old_build["benchmarks"]
        new_benchmarks = new_build["benchmarks"]
        self.assertEqual(
            set(bm["name"] for bm in old_benchmarks),
            set(bm["name"] for bm in new_benchmarks),
        )

    def compare_search_output(
        self, data_prefix_old: pathlib.Path, data_prefix_new: pathlib.Path
    ):
        # Compare the search outputs
        rel_search_dir = pathlib.Path(self.dataset) / "result" / "search"
        old_search_dir = data_prefix_old / rel_search_dir
        new_search_dir = data_prefix_new / rel_search_dir
        self.assertEqual(
            len(list(old_search_dir.glob("*.json"))),
            len(list(new_search_dir.glob("*.json"))),
        )

        for old_search_file in old_search_dir.glob("*.json"):
            new_search_file = (
                data_prefix_new / rel_search_dir / old_search_file.name
            )
            self.assertTrue(new_search_file.exists())

            old_search = json.loads(old_search_file.read_text())
            new_search = json.loads(new_search_file.read_text())

            # Context should be almost the same
            old_context = {**old_search["context"]}
            new_context = {**new_search["context"]}
            for key in ("date", "load_avg"):
                for ctx in [old_context, new_context]:
                    del ctx[key]
            self.assertDictEqual(old_context, new_context)

            # Benchmarks should be almost the same
            old_benchmarks = old_search["benchmarks"]
            new_benchmarks = new_search["benchmarks"]
            self.assertEqual(
                set(bm["name"] for bm in old_benchmarks),
                set(bm["name"] for bm in new_benchmarks),
            )

    def test_run_migration(self):
        data_paths = self.generate_data()
        data_prefix_old = self.temp_dir / "old" / "datasets"
        data_prefix_new = self.temp_dir / "new" / "datasets"
        shutil.copytree(self.data_prefix_common, data_prefix_old)
        shutil.copytree(self.data_prefix_common, data_prefix_new)

        run_build_old(
            data_prefix=data_prefix_old,
            dataset=self.dataset,
            algos=self.algo,
            dataset_config_file=str(data_paths.dataset_configs),
            algo_config_file=str(TEST_DATA_PREFIX / f"{self.algo}.yaml"),
        )
        run_build_new(
            data_prefix=data_prefix_new,
            dataset=self.dataset,
            algos=self.algo,
            dataset_config_file=str(data_paths.dataset_configs),
            algo_config_file=str(TEST_DATA_PREFIX / f"{self.algo}.yaml"),
        )
        self.compare_build_output(
            data_prefix_old=data_prefix_old, data_prefix_new=data_prefix_new
        )

        run_search_old(
            data_prefix=data_prefix_old,
            dataset=self.dataset,
            algos=self.algo,
            dataset_config_file=str(data_paths.dataset_configs),
            algo_config_file=str(TEST_DATA_PREFIX / f"{self.algo}.yaml"),
        )
        run_search_new(
            data_prefix=data_prefix_new,
            dataset=self.dataset,
            algos=self.algo,
            dataset_config_file=str(data_paths.dataset_configs),
            algo_config_file=str(TEST_DATA_PREFIX / f"{self.algo}.yaml"),
        )
        self.compare_search_output(
            data_prefix_old=data_prefix_old, data_prefix_new=data_prefix_new
        )
