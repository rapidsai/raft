#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

source rapids-init-pip

# Delete system libnccl.so to ensure the wheel is used
RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
if [[ ${RAPIDS_CUDA_MAJOR} == "12" ]]; then
  rm -rf /usr/lib64/libnccl*
fi

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
LIBRAFT_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
PYLIBRAFT_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="pylibraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)
RAFT_DASK_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="raft_dask_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)

# echo to expand wildcard before adding `[extra]` requires for pip
rapids-pip-retry install -v \
    "${LIBRAFT_WHEELHOUSE}"/libraft*.whl \
    "${PYLIBRAFT_WHEELHOUSE}"/pylibraft*.whl \
    "$(echo "${RAFT_DASK_WHEELHOUSE}"/raft_dask_"${RAPIDS_PY_CUDA_SUFFIX}"*.whl)[test]"

test_dir="python/raft-dask/raft_dask/tests"

rapids-logger "pytest raft-dask"
python -m pytest --import-mode=append ${test_dir}

rapids-logger "pytest raft-dask (ucx-py only)"
python -m pytest --import-mode=append ${test_dir} --run_ucx

rapids-logger "pytest raft-dask (ucxx only)"
python -m pytest --import-mode=append ${test_dir} --run_ucxx
