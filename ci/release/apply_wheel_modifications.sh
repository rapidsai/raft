#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Usage: bash apply_wheel_modifications.sh <new_version> <cuda_suffix>

VERSION=${1}
CUDA_SUFFIX=${2}

# pyproject.toml versions
sed -i "s/^version = .*/version = \"${VERSION}\"/g" python/pylibraft/pyproject.toml
sed -i "s/^version = .*/version = \"${VERSION}\"/g" python/raft-dask/pyproject.toml

# pylibraft pyproject.toml cuda suffixes
sed -i "s/^name = \"pylibraft\"/name = \"pylibraft${CUDA_SUFFIX}\"/g" python/pylibraft/pyproject.toml
sed -i "s/rmm/rmm${CUDA_SUFFIX}/g" python/pylibraft/pyproject.toml

# raft-dask pyproject.toml cuda suffixes
sed -i "s/^name = \"raft-dask\"/name = \"raft-dask${CUDA_SUFFIX}\"/g" python/raft-dask/pyproject.toml
sed -i "s/pylibraft/pylibraft${CUDA_SUFFIX}/g" python/raft-dask/pyproject.toml
sed -i "s/ucx-py/ucx-py${CUDA_SUFFIX}/g" python/raft-dask/pyproject.toml
