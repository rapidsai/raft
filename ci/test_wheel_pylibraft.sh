#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="libraft_${RAPIDS_PY_CUDA_SUFFIX}" PYTHON_VERSION="3.11" rapids-download-wheels-from-s3 ./dist
RAPIDS_PY_WHEEL_NAME="pylibraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist

artifact_name=$(RAPIDS_PY_WHEEL_NAME="librmm_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_REPOSITORY=rmm RAPIDS_PY_VERSION="3.11" rapids-package-name wheel_python)
commit=$(git ls-remote https://github.com/rapidsai/rmm.git refs/heads/pull-request/1512 | cut -c1-7)
librmm_wheelhouse=$(rapids-get-artifact "ci/rmm/pull-request/1512/${commit}/${artifact_name}")

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install $(echo ./dist/pylibraft*.whl)[test] --find-links ./dist --find-links ${librmm_wheelhouse}

python -m pytest ./python/pylibraft/pylibraft/test
