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

librmm_wheelhouse=$(get-pr-wheel-artifact.sh rmm 1512 cpp)

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install $(echo ./dist/pylibraft*.whl)[test] --find-links ./dist --find-links ${librmm_wheelhouse}

# Run smoke tests for aarch64 pull requests
if [[ "$(arch)" == "aarch64" && "${RAPIDS_BUILD_TYPE}" == "pull-request" ]]; then
    python ./ci/wheel_smoke_test_pylibraft.py
else
    python -m pytest ./python/pylibraft/pylibraft/test
fi
