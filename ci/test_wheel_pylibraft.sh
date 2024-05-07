#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

WHEELHOUSE="${PWD}/dist/"
RAPIDS_PY_WHEEL_NAME="raft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp "${WHEELHOUSE}"
RAPIDS_PY_WHEEL_NAME="raft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python "${WHEELHOUSE}"

python -m pip install "pylibraft-${RAPIDS_PY_CUDA_SUFFIX}[test]>=0.0.0a0" --find-links "${WHEELHOUSE}"

python -m pytest python/pylibraft/pylibraft/test
