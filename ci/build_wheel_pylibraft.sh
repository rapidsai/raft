#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

pyenv global ${RAPIDS_PY_VERSION}

# Set up skbuild options. Enable sccache in skbuild config options
export SKBUILD_CONFIGURE_OPTIONS="-DRAFT_BUILD_WHEELS=ON -DDETECT_CONDA_ENV=OFF -DFIND_RAFT_CPP=OFF -DCMAKE_C_COMPILER_LAUNCHER=/usr/bin/sccache -DCMAKE_CXX_COMPILER_LAUNCHER=/usr/bin/sccache -DCMAKE_CUDA_COMPILER_LAUNCHER=/usr/bin/sccache"

ci/build_wheel.sh pylibraft python/pylibraft
