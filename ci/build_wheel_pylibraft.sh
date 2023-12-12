#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

# Set up skbuild options. Enable sccache in skbuild config options
export SKBUILD_CMAKE_ARGS="-DDETECT_CONDA_ENV=OFF;-DFIND_RAFT_CPP=OFF"

ci/build_wheel.sh pylibraft python/pylibraft
