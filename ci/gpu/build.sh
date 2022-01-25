#!/bin/bash
# Copyright (c) 2020-2022, NVIDIA CORPORATION.
#########################################
# RAFT GPU build and test script for CI #
#########################################

set -e
NUMARGS=$#
ARGS=$*

# Arg parsing function
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Set path and build parallel level
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-8}
export CUDA_REL=${CUDA_VERSION%.*}

# Set home to the job's workspace
export HOME="$WORKSPACE"

# Parse git describei
cd "$WORKSPACE"
export GIT_DESCRIBE_TAG=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`

# Read options for cloning/running downstream repo tests
source "$WORKSPACE/ci/prtest.config"

# ucx-py version
export UCX_PY_VERSION='0.25.*'

################################################################################
# SETUP - Check environment
################################################################################

gpuci_logger "Check environment"
env

gpuci_logger "Check GPU usage"
nvidia-smi

# temporary usage of gpuci_mamba_retry install with packages listed here, looking into
# using the repos yaml files for this
gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids
gpuci_logger "Installing packages needed for RAFT"
gpuci_mamba_retry install -c conda-forge -c rapidsai -c rapidsai-nightly -c nvidia \
      "cudatoolkit=${CUDA_REL}" \
      "libcusolver>=11.2.1" \
      "cudf=${MINOR_VERSION}" \
      "rmm=${MINOR_VERSION}" \
      "breathe" \
      "dask-cudf=${MINOR_VERSION}" \
      "dask-cuda=${MINOR_VERSION}" \
      "ucx-py=${UCX_PY_VERSION}" \
      "rapids-build-env=${MINOR_VERSION}.*" \
      "rapids-notebook-env=${MINOR_VERSION}.*" \
      "rapids-doc-env=${MINOR_VERSION}.*"

# Install the master version of dask, distributed, and dask-ml
gpuci_logger "Install the master version of dask and distributed"
set -x
pip install "git+https://github.com/dask/distributed.git@2022.01.0" --upgrade --no-deps
pip install "git+https://github.com/dask/dask.git@2022.01.0" --upgrade --no-deps
set +x


gpuci_logger "Check compiler versions"
python --version
$CC --version
$CXX --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

################################################################################
# BUILD - Build RAFT tests
################################################################################

gpuci_logger "Adding ${CONDA_PREFIX}/lib to LD_LIBRARY_PATH"

export LD_LIBRARY_PATH_CACHED=$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

gpuci_logger "Build C++ and Python targets"
if hasArg --skip-tests; then
  "$WORKSPACE/build.sh" libraft pyraft libraft -v --compile-libs --nogtest
else
  "$WORKSPACE/build.sh" libraft pyraft libraft -v --compile-libs
fi

gpuci_logger "Building docs"
"$WORKSPACE/build.sh" docs -v

gpuci_logger "Resetting LD_LIBRARY_PATH"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH_CACHED
export LD_LIBRARY_PATH_CACHED=""

################################################################################
# TEST - Run GoogleTest and py.tests for RAFT
################################################################################

if hasArg --skip-tests; then
    gpuci_logger "Skipping Tests"
    exit 0
fi

gpuci_logger "Check GPU usage"
nvidia-smi

gpuci_logger "GoogleTest for raft"
cd "$WORKSPACE/cpp/build"
GTEST_OUTPUT="xml:$WORKSPACE/test-results/raft_cpp/" ./test_raft

gpuci_logger "Python pytest for raft"
cd "$WORKSPACE/python"

python -m pytest --cache-clear --junitxml="$WORKSPACE/junit-raft.xml" -v -s
