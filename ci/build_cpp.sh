#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

# ucx-py version
export UCX_PY_VERSION='0.30.*'

rapids-print-env

rapids-logger "Begin cpp build"

rapids-mamba-retry mambabuild conda/recipes/libraft

rapids-upload-conda-to-s3 cpp
