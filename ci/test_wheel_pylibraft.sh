#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="libraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp ./local-libraft-dep
RAPIDS_PY_WHEEL_NAME="pylibraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./dist

LIBRMM_WHEEL_DIR=$(RAPIDS_PY_WHEEL_NAME="rmm_${RAPIDS_PY_CUDA_SUFFIX}" _rapids-get-pr-artifact rmm 1808 cpp wheel)
PYLIBRMM_WHEEL_DIR=$(RAPIDS_PY_WHEEL_NAME="rmm_${RAPIDS_PY_CUDA_SUFFIX}" _rapids-get-pr-artifact rmm 1808 python wheel)
echo "librmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${LIBRMM_WHEEL_DIR}/librmm_*.whl)" >> /tmp/constraints.txt
echo "rmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${PYLIBRMM_WHEEL_DIR}/rmm_*.whl)" >> /tmp/constraints.txt
export PIP_CONSTRAINT="/tmp/constraints.txt"


# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install \
    ./local-libraft-dep/libraft*.whl \
    "$(echo ./dist/pylibraft*.whl)[test]"

python -m pytest ./python/pylibraft/pylibraft/tests
