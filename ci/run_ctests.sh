#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

if [ -d "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/libraft/" ]; then
    # Support customizing the ctests' install location
    cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/libraft/"
    ctest --output-on-failure "$@"
fi
