#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
RAPIDS_PY_WHEEL_NAME="libraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp ./local-libraft-dep
RAPIDS_PY_WHEEL_NAME="pylibraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./dist


# echo to expand wildcard before adding `[extra]` requires for pip
rapids-pip-retry install \
    ./local-libraft-dep/libraft*.whl \
    "$(echo ./dist/pylibraft*.whl)[test]"

python -m pytest ./python/pylibraft/pylibraft/tests
