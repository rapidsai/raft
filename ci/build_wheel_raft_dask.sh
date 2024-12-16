#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/raft-dask"

# Set up skbuild options. Enable sccache in skbuild config options
export SKBUILD_CMAKE_ARGS="-DDETECT_CONDA_ENV=OFF;-DFIND_RAFT_CPP=OFF"

ci/build_wheel.sh raft-dask ${package_dir} python
ci/validate_wheel.sh ${package_dir} final_dist raft-dask
