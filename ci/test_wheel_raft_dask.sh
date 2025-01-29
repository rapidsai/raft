#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="libraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp ./local-libraft-dep
RAPIDS_PY_WHEEL_NAME="pylibraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./local-pylibraft-dep
RAPIDS_PY_WHEEL_NAME="raft_dask_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./dist

# TODO: remove me once gha-tools is released
if ! command -v "rapids-pip-retry" &> /dev/null
then
    git clone --branch rapids-pip-retry https://github.com/gforsyth/gha-tools.git

    export PATH="$PWD/gha-tools/tools":$PATH
fi
# echo to expand wildcard before adding `[extra]` requires for pip
rapids-pip-retry install -v \
    ./local-libraft-dep/libraft*.whl \
    ./local-pylibraft-dep/pylibraft*.whl \
    "$(echo ./dist/raft_dask_${RAPIDS_PY_CUDA_SUFFIX}*.whl)[test]"

test_dir="python/raft-dask/raft_dask/tests"

rapids-logger "pytest raft-dask"
python -m pytest --import-mode=append ${test_dir}

rapids-logger "pytest raft-dask (ucx-py only)"
python -m pytest --import-mode=append ${test_dir} --run_ucx

rapids-logger "pytest raft-dask (ucxx only)"
python -m pytest --import-mode=append ${test_dir} --run_ucxx
