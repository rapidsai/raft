#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="raft_dask_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist

# Download the pylibraft built in the previous step
RAPIDS_PY_WHEEL_NAME="pylibraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./local-pylibraft-dep
python -m pip install --no-deps ./local-pylibraft-dep/pylibraft*.whl

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install $(echo ./dist/raft_dask*.whl)[test]

python -m pytest ./python/raft-dask/raft_dask/test
python -m pytest ./python/raft-dask/raft_dask/test --run_ucx
python -m pytest ./python/raft-dask/raft_dask/test --run_ucxx
