#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin py build"

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

version=$(rapids-generate-version)
git_commit=$(git rev-parse HEAD)
export RAPIDS_PACKAGE_VERSION=${version} 
echo "${version}" | tr -d '"' > VERSION

package_dir="python"
for package_name in pylibraft raft-dask; do 
  underscore_package_name=$(echo "${package_name}" | tr "-" "_")
  sed -i "/^__git_commit__/ s/= .*/= \"${git_commit}\"/g" "${package_dir}/${package_name}/${underscore_package_name}/_version.py"
done

# TODO: Remove `--no-test` flags once importing on a CPU
# node works correctly
rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  conda/recipes/pylibraft

rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
  conda/recipes/raft-dask

# Build ann-bench for each cuda and python version
version_file_raft_ann_bench="python/raft-ann-bench/src/raft-ann-bench/_version.py"
sed -i "/^__git_commit__/ s/= .*/= \"${git_commit}\"/g" ${version_file_raft_dask}
rapids-conda-retry mambabuild \
--no-test \
--channel "${CPP_CHANNEL}" \
--channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
conda/recipes/raft-ann-bench

# Build ann-bench-cpu only in CUDA 11 jobs since it only depends on python
# version
RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
if [[ ${RAPIDS_CUDA_MAJOR} == "11" ]]; then
  rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
  conda/recipes/raft-ann-bench-cpu
fi

rapids-upload-conda-to-s3 python
