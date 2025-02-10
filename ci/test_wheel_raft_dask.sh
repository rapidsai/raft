#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
RAPIDS_PY_WHEEL_NAME="libraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp ./local-libraft-dep
RAPIDS_PY_WHEEL_NAME="pylibraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./local-pylibraft-dep
RAPIDS_PY_WHEEL_NAME="raft_dask_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./dist

LIBRMM_WHEEL_DIR=$(RAPIDS_PY_WHEEL_NAME="rmm_${RAPIDS_PY_CUDA_SUFFIX}" _rapids-get-pr-artifact rmm 1808 cpp wheel)
PYLIBRMM_WHEEL_DIR=$(RAPIDS_PY_WHEEL_NAME="rmm_${RAPIDS_PY_CUDA_SUFFIX}" _rapids-get-pr-artifact rmm 1808 python wheel)
LIBUCXX_WHEEL_DIR=$(RAPIDS_PY_WHEEL_NAME="libucxx_${RAPIDS_PY_CUDA_SUFFIX}" _rapids-get-pr-artifact ucxx 364 cpp wheel)
PYLIBUCXX_WHEEL_DIR=$(RAPIDS_PY_WHEEL_NAME="ucxx_${RAPIDS_PY_CUDA_SUFFIX}" _rapids-get-pr-artifact ucxx 364 python wheel)
echo "librmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${LIBRMM_WHEEL_DIR}/librmm_*.whl)" >> /tmp/constraints.txt
echo "rmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${PYLIBRMM_WHEEL_DIR}/rmm_*.whl)" >> /tmp/constraints.txt
echo "libucxx-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${LIBUCXX_WHEEL_DIR}/libucxx_*.whl)" >> /tmp/constraints.txt
echo "ucxx-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${PYLIBUCXX_WHEEL_DIR}/ucxx_*.whl)" >> /tmp/constraints.txt
export PIP_CONSTRAINT="/tmp/constraints.txt"

# echo to expand wildcard before adding `[extra]` requires for pip
rapids-pip-retry install -v \
    ./local-libraft-dep/libraft*.whl \
    ./local-pylibraft-dep/pylibraft*.whl \
    "$(echo ./dist/raft_dask_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]"

test_dir="python/raft-dask/raft_dask/tests"

rapids-logger "pytest raft-dask"
python -m pytest --import-mode=append ${test_dir}

rapids-logger "pytest raft-dask (ucx-py only)"
python -m pytest --import-mode=append ${test_dir} --run_ucx

rapids-logger "pytest raft-dask (ucxx only)"
python -m pytest --import-mode=append ${test_dir} --run_ucxx
