#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION.
#########################################
# cuML GPU build and test script for CI #
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
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}
export CUDA_REL=${CUDA_VERSION%.*}

# Set home to the job's workspace
export HOME="$WORKSPACE"

# Parse git describei
cd "$WORKSPACE"
export GIT_DESCRIBE_TAG=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`

# Read options for cloning/running downstream repo tests
source "$WORKSPACE/ci/prtest.config"

################################################################################
# SETUP - Check environment
################################################################################

gpuci_logger "Check environment"
env

gpuci_logger "Check GPU usage"
nvidia-smi

# temporary usage of gpuci_conda_retry install with packages listed here, looking into
# using the repos yaml files for this
gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids
gpuci_logger "Installing packages needed for RAFT"
gpuci_conda_retry install -c conda-forge -c rapidsai -c rapidsai-nightly -c nvidia \
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

if [ "$RUN_CUML_LIBCUML_TESTS" = "ON" ] || [ "$RUN_CUML_PRIMS_TESTS" = "ON" ] || [ "$RUN_CUML_PYTHON_TESTS" = "ON" ]; then
  gpuci_conda_retry install -c conda-forge -c rapidsai -c rapidsai-nightly -c nvidia \
      "nvstrings=${MINOR_VERSION}" \
      "libcumlprims=${MINOR_VERSION}" \
      "lapack" \
      "umap-learn" \
      "nccl>=2.5" \
      "statsmodels" \
      "xgboost====1.0.2dev.rapidsai0.13" \
      "lightgbm"
fi

if [ "$RUN_CUGRAPH_LIBCUGRAPH_TESTS" = "ON" ] || [ "$RUN_CUGRAPH_PYTHON_TESTS" = "ON" ]; then
  gpuci_conda_retry install -c nvidia -c rapidsai -c rapidsai-nightly -c conda-forge -c defaults \
      "networkx>=2.3" \
      "python-louvain" \
      "libcypher-parser" \
      "ipython=7.3*" \
      "jupyterlab"
fi

# Install the master version of dask, distributed, and dask-ml
gpuci_logger "pip install git+https://github.com/dask/distributed.git --upgrade --no-deps"
pip install "git+https://github.com/dask/distributed.git" --upgrade --no-deps
gpuci_logger "pip install git+https://github.com/dask/dask.git --upgrade --no-deps"
pip install "git+https://github.com/dask/dask.git" --upgrade --no-deps


gpuci_logger "Check versions"
python --version
$CC --version
$CXX --version
conda info
conda config --show-sources
conda list --show-channel-urls


################################################################################
# BUILD - Build RAFT tests
################################################################################

gpuci_logger "Adding ${CONDA_PREFIX}/lib to LD_LIBRARY_PATH"

export LD_LIBRARY_PATH_CACHED=$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

gpuci_logger "Build libcuml, cuml, prims and bench targets"
"$WORKSPACE/build.sh" cppraft pyraft -v

gpuci_logger "Resetting LD_LIBRARY_PATH"

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH_CACHED
export LD_LIBRARY_PATH_CACHED=""

gpuci_logger "Build treelite for GPU testing"

cd "$WORKSPACE"


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
GTEST_OUTPUT="xml:"$WORKSPACE/test-results/raft_cpp/" ./test/ml

gpuci_logger "Python pytest for cuml"
cd "$WORKSPACE/python"

pytest --cache-clear --junitxml="$WORKSPACE/junit-cuml.xml" -v -s


################################################################################
# cuML CI
################################################################################

if [ "$RUN_CUML_LIBCUML_TESTS" = "ON" ] || [ "$RUN_CUML_PRIMS_TESTS" = "ON" ] || [ "$RUN_CUML_PYTHON_TESTS" = "ON" ] || [ "$RUN_CUGRAPH_LIBCUGRAPH_TESTS" = "ON" ] || [ "$RUN_CUGRAPH_PYTHON_TESTS" = "ON" ]; then
  cd "$WORKSPACE"
  mkdir "$WORKSPACE/test_downstream_repos"
  cd "$WORKSPACE/test_downstream_repos"
  export RAFT_PATH="$WORKSPACE"
fi

if [ "$RUN_CUML_LIBCUML_TESTS" = "ON" ] || [ "$RUN_CUML_PRIMS_TESTS" = "ON" ] || [ "$RUN_CUML_PYTHON_TESTS" = "ON" ]; then
  cd "$WORKSPACE/test_downstream_repos"

  ## Change fork and branch to be tested here:
  git clone https://github.com/rapidsai/cuml.git -b branch-0.14


  ## Build cuML and run tests, uncomment the tests you want to run
  "$WORKSPACE/test_downstream_repos/cuml/build.sh"

  if [ "$RUN_CUML_LIBCUML_TESTS" = "ON" ]; then
    gpuci_logger "GoogleTest for libcuml"
    cd "$WORKSPACE/cpp/build"
    GTEST_OUTPUT="xml:$WORKSPACE/test-results/libcuml_cpp/" ./test/ml
  fi

  if [ "$RUN_CUML_PYTHON_TESTS" = "ON" ]; then
    gpuci_logger "Python pytest for cuml"
    cd "$WORKSPACE/python"
    pytest --cache-clear --junitxml="$WORKSPACE/junit-cuml.xml" -v -s -m "not memleak"
  fi

  if [ "$RUN_CUML_PRIMS_TESTS" = "ON" ]; then
    gpuci_logger "Run ml-prims test"
    cd "$WORKSPACE/cpp/build"
    GTEST_OUTPUT="xml:$WORKSPACE/test-results/prims/ ./test/prims
  fi
fi


################################################################################
# cuGraph CI
################################################################################

if [ "$RUN_CUGRAPH_LIBCUGRAPH_TESTS" = "ON" ] || [ "$RUN_CUGRAPH_PYTHON_TESTS" = "ON" ]; then
  cd "$WORKSPACE/test_downstream_repos"

  ## Change fork and branch to be tested here:
  git clone https://github.com/rapidsai/cugraph.git -b branch-0.14

  "$WORKSPACE/test_downstream_repos/cugraph/build.sh" clean libcugraph cugraph

  if [ "$RUN_CUGRAPH_LIBCUGRAPH_TESTS" = "ON" ]; then
    gpuci_logger "GoogleTest for libcugraph"
    cd "$WORKSPACE/cpp/build"
    "$WORKSPACE/ci/test.sh" ${TEST_MODE_FLAG} | tee testoutput.txt
  fi

  if [ "$RUN_CUGRAPH_PYTHON_TESTS" = "ON" ]; then
    gpuci_logger "Python pytest for cugraph"
    cd "$WORKSPACE/python"
  fi
fi
