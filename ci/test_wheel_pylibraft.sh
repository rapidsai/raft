#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="pylibraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist

LIBRMM_CHANNEL=$(
  RAPIDS_PY_WHEEL_NAME=rmm_${RAPIDS_PY_CUDA_SUFFIX} rapids-get-pr-wheel-artifact rmm 1678 cpp
)
RMM_CHANNEL=$(
  RAPIDS_PY_WHEEL_NAME=rmm_${RAPIDS_PY_CUDA_SUFFIX} rapids-get-pr-wheel-artifact rmm 1678 python
)

echo "rmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${RMM_CHANNEL}/rmm*.whl)" > /tmp/constraints.txt
echo "librmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${LIBRMM_CHANNEL}/librmm*.whl)" >> /tmp/constraints.txt
echo "" >> /tmp/constraints.txt
export PIP_CONSTRAINT=/tmp/constraints.txt

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install $(echo ./dist/pylibraft*.whl)[test]

python -m pytest ./python/pylibraft/pylibraft/test
