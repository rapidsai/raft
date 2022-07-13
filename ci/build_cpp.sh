#!/bin/bash
set -e

# Update env vars
source rapids-env-update

# Check environment
source ci/check_environment.sh

# Use Ninja to build
export CMAKE_GENERATOR="Ninja"

# ucx-py version
export UCX_PY_VERSION='0.27.*'

export CUDAHOSTCXX=${CXX}
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}

if [[ "$BUILD_MODE" = "branch" && "$SOURCE_BRANCH" = branch-* ]] ; then
  export VERSION_SUFFIX=$(date +%y%m%d)
else
  export VERSION_SUFFIX=""
fi

################################################################################
# BUILD - Conda package builds (LIBCUGRAPH)
################################################################################
gpuci_logger "Begin cpp build"

gpuci_mamba_retry mambabuild \
  --croot /tmp/conda-bld-workspace \
  --output-folder /tmp/conda-bld-output \
  conda/recipes/libraft

rapids-upload-conda-to-s3 cpp
