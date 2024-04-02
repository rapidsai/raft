#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

if [[ ! -d "/tmp/gha-tools" ]]; then
    git clone https://github.com/msarahan/gha-tools.git -b get-pr-wheel-artifact /tmp/gha-tools
    export PATH="/tmp/gha-tools/tools:${PATH}"
fi

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="libraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp ./dist
RAPIDS_PY_WHEEL_NAME="pylibraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./dist

RAPIDS_PY_WHEEL_NAME="librmm_${RAPIDS_PY_CUDA_SUFFIX}" librmm_wheelhouse=$(rapids-get-pr-wheel-artifact rmm 1512 cpp)

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install $(echo ./dist/pylibraft*.whl)[test] --find-links ./dist --find-links ${librmm_wheelhouse}

python -m pytest ./python/pylibraft/pylibraft/test
