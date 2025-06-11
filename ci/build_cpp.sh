#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

set -euo pipefail

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin cpp build"

sccache --zero-stats

RAPIDS_PACKAGE_VERSION=$(rapids-generate-version)
export RAPIDS_PACKAGE_VERSION

RAPIDS_ARTIFACTS_DIR=${RAPIDS_ARTIFACTS_DIR:-"${PWD}/artifacts"}
mkdir -p "${RAPIDS_ARTIFACTS_DIR}"
export RAPIDS_ARTIFACTS_DIR

# populates `RATTLER_CHANNELS` array and `RATTLER_ARGS` array
source rapids-rattler-channel-string

# --no-build-id allows for caching with `sccache`
# more info is available at
# https://rattler.build/latest/tips_and_tricks/#using-sccache-or-ccache-with-rattler-build
rattler-build build --recipe conda/recipes/libraft \
                    "${RATTLER_ARGS[@]}" \
                    "${RATTLER_CHANNELS[@]}"

sccache --show-adv-stats

# remove build_cache directory
rm -rf "$RAPIDS_CONDA_BLD_OUTPUT_DIR"/build_cache
