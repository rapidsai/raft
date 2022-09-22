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
CONDA_ARTIFACT_PATH=${WORKSPACE}/ci/artifacts/raft/cpu/.conda-bld/ # notice there is no `linux-64` here

# Set home to the job's workspace
export HOME=$WORKSPACE

# Parse git describe
cd $WORKSPACE
export GIT_DESCRIBE_TAG=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`
unset GIT_DESCRIBE_TAG

# ucx-py version
export UCX_PY_VERSION='0.28.*'

################################################################################
# SETUP - Check environment
################################################################################

gpuci_logger "Check environment"
env

gpuci_logger "Check GPU usage"
nvidia-smi

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

# Install pre-built conda packages from previous CI step
gpuci_logger "Install libraft conda packages from CPU job"
gpuci_mamba_retry install -y -c "${CONDA_ARTIFACT_PATH}" libraft-headers libraft-distance libraft-nn libraft-tests

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

################################################################################
# BUILD - Build RAFT tests
################################################################################

gpuci_logger "Build and install Python targets"
CONDA_BLD_DIR="$WORKSPACE/.conda-bld"
gpuci_mamba_retry install boa

# Install pylibraft first since it's a dependency of raft-dask
gpuci_conda_retry mambabuild --no-build-id --croot "${CONDA_BLD_DIR}" conda/recipes/pylibraft -c "${CONDA_ARTIFACT_PATH}" --python="${PYTHON}"
gpuci_mamba_retry install -y -c "${CONDA_BLD_DIR}" -c "${CONDA_ARTIFACT_PATH}" pylibraft

gpuci_conda_retry mambabuild --no-build-id --croot "${CONDA_BLD_DIR}" conda/recipes/raft-dask -c "${CONDA_ARTIFACT_PATH}" --python="${PYTHON}"
gpuci_mamba_retry install -y -c "${CONDA_BLD_DIR}" -c "${CONDA_ARTIFACT_PATH}" raft-dask

################################################################################
# TEST - Run GoogleTest and py.tests for RAFT
################################################################################

if hasArg --skip-tests; then
    gpuci_logger "Skipping Tests"
    exit 0
fi

# Install the master version of dask, distributed, and dask-ml
gpuci_logger "Install the master version of dask and distributed"
set -x
pip install "git+https://github.com/dask/distributed.git@main" --upgrade --no-deps
pip install "git+https://github.com/dask/dask.git@main" --upgrade --no-deps
set +x

gpuci_logger "Check GPU usage"
nvidia-smi

gpuci_logger "GoogleTest for libraft"
GTEST_OUTPUT="xml:${WORKSPACE}/test-results/raft_cpp/" $CONDA_PREFIX/bin/libraft/gtests/test_raft

gpuci_logger "Python pytest for pylibraft"
cd "$WORKSPACE/python/pylibraft/pylibraft/test"
pytest --cache-clear --junitxml="$WORKSPACE/junit-pylibraft.xml" -v -s

gpuci_logger "Python pytest for raft-dask"
cd "$WORKSPACE/python/raft-dask/raft_dask/test"
pytest --cache-clear --junitxml="$WORKSPACE/junit-raft-dask.xml" -v -s

if [ "$(arch)" = "x86_64" ]; then
  gpuci_logger "Building docs"
  gpuci_mamba_retry install "rapids-doc-env=${MINOR_VERSION}.*"
  "$WORKSPACE/build.sh" docs -v
fi
