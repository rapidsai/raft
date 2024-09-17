#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="raft_dask_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist

LIBRMM_CHANNEL=$(rapids-get-artifact "ci/rmm/pull-request/1678/4127d9c/rmm_wheel_cpp_rmm_${RAPIDS_PY_CUDA_SUFFIX}_x86_64.tar.gz")
RMM_CHANNEL=$(rapids-get-artifact "ci/rmm/pull-request/1678/4127d9c/rmm_wheel_python_rmm_${RAPIDS_PY_CUDA_SUFFIX}_py${RAPIDS_PY_VERSION//./}_x86_64.tar.gz")

echo "rmm-${RAPIDS_PY_CUDA_SUFFIX} @ $(echo ${RMM_CHANNEL}/rmm*.whl)" > /tmp/constraints.txt
echo "librmm-${RAPIDS_PY_CUDA_SUFFIX} @ $(echo ${LIBRMM_CHANNEL}/librmm*.whl)" >> /tmp/constraints.txt
echo "" >> /tmp/constraints.txt
export PIP_CONSTRAINT=/tmp/constraints.txt

# Download the pylibraft built in the previous step
RAPIDS_PY_WHEEL_NAME="pylibraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./local-pylibraft-dep
python -m pip install --no-deps ./local-pylibraft-dep/pylibraft*.whl

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install -v "$(echo ./dist/raft_dask_${RAPIDS_PY_CUDA_SUFFIX}*.whl)[test]"

test_dir="python/raft-dask/raft_dask/test"

rapids-logger "pytest raft-dask"
python -m pytest --import-mode=append ${test_dir}

rapids-logger "pytest raft-dask (ucx-py only)"
python -m pytest --import-mode=append ${test_dir} --run_ucx

rapids-logger "pytest raft-dask (ucxx only)"
python -m pytest --import-mode=append ${test_dir} --run_ucxx
