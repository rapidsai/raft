#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin cpp build"

LIBRMM_CHANNEL=$(rapids-get-pr-conda-artifact rmm 1678 cpp)
RMM_CHANNEL=$(rapids-get-pr-conda-artifact rmm 1678 python)
UCXX_CHANNEL=$(rapids-get-pr-conda-artifact ucxx 278 cpp)

RAPIDS_PACKAGE_VERSION=$(rapids-generate-version) rapids-conda-retry mambabuild \
    --channel "${LIBRMM_CHANNEL}" \
    --channel "${RMM_CHANNEL}" \
    --channel "${UCXX_CHANNEL}" \
    conda/recipes/libraft

rapids-upload-conda-to-s3 cpp
