#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="raft_dask_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist
ucxx_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="ucxx_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact ucxx 226 python)
distributed_ucxx_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="distributed_ucxx_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact ucxx 226 python)


# Download the pylibraft built in the previous step
RAPIDS_PY_WHEEL_NAME="pylibraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./local-pylibraft-dep
python -m pip install --no-deps ./local-pylibraft-dep/pylibraft*.whl

python -m pip install "raft_dask-${RAPIDS_PY_CUDA_SUFFIX}[test]>=0.0.0a0" --find-links dist/ --find-links "${ucxx_wheelhouse}" --find-links "${distributed_ucxx_wheelhouse}"

python -m pytest ./python/raft-dask/raft_dask/test
python -m pytest ./python/raft-dask/raft_dask/test --run_ucx
python -m pytest ./python/raft-dask/raft_dask/test --run_ucxx
