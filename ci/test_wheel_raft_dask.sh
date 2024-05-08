#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

WHEELHOUSE="${PWD}/dist/"
RAPIDS_PY_WHEEL_NAME="raft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp "${WHEELHOUSE}"
RAPIDS_PY_WHEEL_NAME="raft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python "${WHEELHOUSE}"

python -m pip install "raft-dask-${RAPIDS_PY_CUDA_SUFFIX}[test]>=0.0.0a0" --find-links "${WHEELHOUSE}"

test_dir="python/raft-dask/raft_dask/test"

# rapids-logger "pytest raft-dask"
# python -m pytest ${test_dir}

# rapids-logger "pytest raft-dask (ucx-py only)"
# python -m pytest ${test_dir} --run_ucx

rapids-logger "pytest raft-dask (ucxx only)"
python -m pytest ${test_dir} --run_ucxx
