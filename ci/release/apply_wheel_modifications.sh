#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Usage: bash apply_wheel_modifications.sh <new_version> <cuda_suffix>

VERSION=${1}
CUDA_SUFFIX=${2}

# __init__.py versions
sed -i "s/__version__ = .*/__version__ = \"${VERSION}\"/g" python/pylibraft/pylibraft/__init__.py
sed -i "s/__version__ = .*/__version__ = \"${VERSION}\"/g" python/raft-dask/raft_dask/__init__.py

# setup.py versions
sed -i "s/version=.*,/version=\"${VERSION}\",/g" python/pylibraft/pylibraft/__init__.py
sed -i "s/version=.*,/version=\"${VERSION}\",/g" python/raft-dask/raft_dask/__init__.py

# pylibraft setup.py cuda suffixes
sed -i "s/name=\"pylibraft\"/name=\"pylibraft${CUDA_SUFFIX}\"/g" python/pylibraft/setup.py
sed -i "s/rmm/rmm${CUDA_SUFFIX}/g" python/pylibraft/setup.py

# raft-dask setup.py cuda suffixes
sed -i "s/name=\"raft-dask\"/name=\"raft-dask${CUDA_SUFFIX}\"/g" python/raft-dask/setup.py
sed -i "s/ucx-py/ucx-py${CUDA_SUFFIX}/g" python/raft-dask/setup.py
sed -i "s/pylibraft/pylibraft${CUDA_SUFFIX}/g" python/raft-dask/setup.py

# Dependency versions in pyproject.toml
sed -i "s/rmm/rmm${CUDA_SUFFIX}/g" python/pylibraft/pyproject.toml
