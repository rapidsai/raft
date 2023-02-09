#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
#########################################
#  RAFT CPU conda build script for CI   #
#########################################
set -e

# Set path and build parallel level
# openmpi dir is required on CentOS for finding MPI libs from cmake
if [[ -e /etc/os-release ]] && (grep -qi centos /etc/os-release); then
    export PATH=/opt/conda/bin:/usr/local/cuda/bin:/usr/lib64/openmpi/bin:$PATH
else
    export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
fi
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-8}

# Set home to the job's workspace
export HOME=$WORKSPACE

# Switch to project root; also root of repo checkout
cd $WORKSPACE

# If nightly build, append current YYMMDD to version
if [[ "$BUILD_MODE" = "branch" && "$SOURCE_BRANCH" = branch-* ]] ; then
  export VERSION_SUFFIX=$(date +%y%m%d)
else
  export VERSION_SUFFIX=""
fi

# Setup 'gpuci_conda_retry' for build retries (results in 2 total attempts)
export GPUCI_CONDA_RETRY_MAX=1
export GPUCI_CONDA_RETRY_SLEEP=30

# Workaround to keep Jenkins builds working
# until we migrate fully to GitHub Actions
export RAPIDS_CUDA_VERSION="${CUDA}"
export SCCACHE_BUCKET=rapids-sccache
export SCCACHE_REGION=us-west-2
export SCCACHE_IDLE_TIMEOUT=32768

# Use Ninja to build
export CMAKE_GENERATOR="Ninja"
export CONDA_BLD_DIR="${WORKSPACE}/.conda-bld"

# ucx-py version
export UCX_PY_VERSION='0.31.*'

################################################################################
# SETUP - Check environment
################################################################################

gpuci_logger "Check environment variables"
env

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

# Remove rapidsai-nightly channel if we are building main branch
if [ "$SOURCE_BRANCH" = "main" ]; then
  conda config --system --remove channels rapidsai-nightly
fi

gpuci_logger "Check versions"
python --version
$CC --version
$CXX --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

if [ "$BUILD_LIBRAFT" == "1" ]; then
  # If we are doing CUDA builds, libraft package is located at ${CONDA_BLD_DIR}
  CONDA_LOCAL_CHANNEL="${CONDA_BLD_DIR}"
else
  # If we are doing Python builds only, libraft package is placed here by Project Flash
  CONDA_LOCAL_CHANNEL="ci/artifacts/raft/cpu/.conda-bld/"
fi

gpuci_mamba_retry install -c conda-forge boa

###############################################################################
# BUILD - Conda package builds
###############################################################################

if [ "$BUILD_LIBRAFT" == "1" ]; then
  gpuci_logger "Building conda packages for libraft-nn, libraft-distance, libraft-headers and libraft-tests"
  if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    gpuci_conda_retry mambabuild --no-build-id --croot ${CONDA_BLD_DIR} conda/recipes/libraft
  else
    gpuci_conda_retry mambabuild --no-build-id --croot ${CONDA_BLD_DIR} --dirty --no-remove-work-dir conda/recipes/libraft
    gpuci_logger "`ls ${CONDA_BLD_DIR}/work`"
    mkdir -p ${CONDA_BLD_DIR}/libraft/work
    mv ${CONDA_BLD_DIR}/work ${CONDA_BLD_DIR}/libraft/work
  fi
  sccache --show-stats
else
  gpuci_logger "SKIPPING build of conda packages for libraft-nn, libraft-distance, libraft-headers and libraft-tests"

  # Install pre-built conda packages from previous CI step
  gpuci_logger "Install libraft conda packages from CPU job"
  CONDA_ARTIFACT_PATH=${WORKSPACE}/ci/artifacts/raft/cpu/.conda-bld/ # notice there is no `linux-64` here
  gpuci_mamba_retry install -y -c ${CONDA_ARTIFACT_PATH} libraft-headers libraft-distance libraft-nn libraft-tests
fi

if [ "$BUILD_RAFT" == '1' ]; then
  gpuci_logger "Building Python conda packages for raft"
  if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    gpuci_conda_retry mambabuild --no-build-id --croot ${CONDA_BLD_DIR} conda/recipes/pylibraft --python=$PYTHON
    gpuci_conda_retry mambabuild --no-build-id --croot ${CONDA_BLD_DIR} conda/recipes/raft-dask --python=$PYTHON
  else
    gpuci_conda_retry mambabuild --no-build-id --croot ${CONDA_BLD_DIR} conda/recipes/pylibraft -c ${CONDA_LOCAL_CHANNEL} --dirty --no-remove-work-dir --python=$PYTHON
    mkdir -p ${CONDA_BLD_DIR}/pylibraft/work
    mv ${CONDA_BLD_DIR}/work ${CONDA_BLD_DIR}/pylibraft/work

    gpuci_conda_retry mambabuild --no-build-id --croot ${CONDA_BLD_DIR} conda/recipes/raft-dask -c ${CONDA_LOCAL_CHANNEL} --dirty --no-remove-work-dir --python=$PYTHON
    mkdir -p ${CONDA_BLD_DIR}/raft-dask/work
    mv ${CONDA_BLD_DIR}/work ${CONDA_BLD_DIR}/raft-dask/work
  fi
else
  gpuci_logger "SKIPPING build of Python conda packages for raft"
fi

################################################################################
# UPLOAD - Conda packages
################################################################################

# Uploads disabled due to new GH Actions implementation
# gpuci_logger "Upload conda packages"
# source ci/cpu/upload.sh
