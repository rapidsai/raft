#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

# If nightly build, append current YYMMDD to version
if [[ "$RAPIDS_BUILD_TYPE" = "nightly" && "$RAPIDS_REF_NAME" = branch-* ]] ; then
  export VERSION_SUFFIX=$(date +%y%m%d)
else
  export VERSION_SUFFIX=""
fi

# ucx-py version
export UCX_PY_VERSION='0.30.*'

rapids-print-env

rapids-logger "Begin py build"

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

# TODO: Remove `--no-test` flag once importing on a CPU
# node works correctly
rapids-mamba-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  conda/recipes/pylibraft

# TODO: Remove `--no-test` flag once importing on a CPU
# node works correctly
rapids-mamba-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  conda/recipes/raft-dask

rapids-upload-conda-to-s3 python
