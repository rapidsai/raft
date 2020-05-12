#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION.
#########################################
# cuML GPU build and test script for CI #
#########################################

set -e
NUMARGS=$#
ARGS=$*

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Arg parsing function
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Set path and build parallel level
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=4
export CUDA_REL=${CUDA_VERSION%.*}

# Set home to the job's workspace
export HOME=$WORKSPACE

# Parse git describei
cd $WORKSPACE
export GIT_DESCRIBE_TAG=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`

# Read options for cloning/running downstream repo tests
source $WORKSPACE/ci/prtest.config

################################################################################
# SETUP - Check environment
################################################################################

logger "Check environment..."
env

logger "Check GPU usage..."
nvidia-smi

# temporary usage of conda install with packages listed here, looking into
# using the repos yaml files for this
logger "Activate conda env..."
source activate gdf
logger "Installing packages needed for RAFT..."
conda install -c conda-forge -c rapidsai -c rapidsai-nightly -c nvidia \
      "cupy>=7,<8.0.0a0" \
      "cudatoolkit=${CUDA_REL}" \
      "cudf=${MINOR_VERSION}" \
      "rmm=${MINOR_VERSION}" \
      "cmake==3.14.3" \
      "nccl>=2.5" \
      "dask>=2.12.0" \
      "distributed>=2.12.0" \
      "dask-cudf=${MINOR_VERSION}" \
      "dask-cuda=${MINOR_VERSION}" \
      "ucx-py=${MINOR_VERSION}"

# Install the master version of dask, distributed, and dask-ml
logger "pip install git+https://github.com/dask/distributed.git --upgrade --no-deps"
pip install "git+https://github.com/dask/distributed.git" --upgrade --no-deps
logger "pip install git+https://github.com/dask/dask.git --upgrade --no-deps"
pip install "git+https://github.com/dask/dask.git" --upgrade --no-deps


logger "Check versions..."
python --version
$CC --version
$CXX --version
conda list

################################################################################
# BUILD - Build RAFT tests
################################################################################

logger "Adding ${CONDA_PREFIX}/lib to LD_LIBRARY_PATH"

export LD_LIBRARY_PATH_CACHED=$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

logger "Build C++ and Python targets..."
$WORKSPACE/build.sh cppraft pyraft -v

logger "Resetting LD_LIBRARY_PATH..."

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH_CACHED
export LD_LIBRARY_PATH_CACHED=""



################################################################################
# TEST - Run GoogleTest and py.tests for RAFT
################################################################################

if hasArg --skip-tests; then
    logger "Skipping Tests..."
    exit 0
fi

logger "Check GPU usage..."
nvidia-smi

logger "GoogleTest for raft..."
cd $WORKSPACE/cpp/build
GTEST_OUTPUT="xml:${WORKSPACE}/test-results/raft_cpp/" ./test_raft

logger "Python pytest for cuml..."
cd $WORKSPACE/python

python -m pytest --cache-clear --junitxml=${WORKSPACE}/junit-cuml.xml -v -s
