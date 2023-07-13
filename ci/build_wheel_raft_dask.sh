#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

pyenv global ${RAPIDS_PY_VERSION}

# Set up skbuild options. Enable sccache in skbuild config options
export SKBUILD_CONFIGURE_OPTIONS="-DRAFT_BUILD_WHEELS=ON -DDETECT_CONDA_ENV=OFF -DFIND_RAFT_CPP=OFF"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

git clone https://github.com/divyegala/gha-tools.git -b wheel-local-runs /tmp/gha-tools

RAPIDS_PY_WHEEL_NAME=pylibraft_${RAPIDS_PY_CUDA_SUFFIX} /tmp/gha-tools/tools/rapids-download-wheels-from-s3 ./local-pylibraft && python -m pip install --no-deps ./local-pylibraft/pylibraft*.whl

ci/build_wheel.sh raft_dask python/raft-dask
