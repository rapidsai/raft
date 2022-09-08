#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Adopted from https://github.com/tmcdonell/travis-scripts/blob/dfaac280ac2082cd6bcaba3217428347899f2975/update-accelerate-buildbot.sh

set -e

# Setup 'gpuci_retry' for upload retries (results in 4 total attempts)
export GPUCI_RETRY_MAX=3
export GPUCI_RETRY_SLEEP=30

# Set label option.
#LABEL_OPTION="--label testing"
LABEL_OPTION="--label main"

# Skip uploads unless BUILD_MODE == "branch"
if [ ${BUILD_MODE} != "branch" ]; then
  echo "Skipping upload"
  return 0
fi

# Skip uploads if there is no upload key
if [ -z "$MY_UPLOAD_KEY" ]; then
  echo "No upload key"
  return 0
fi

################################################################################
# UPLOAD - Conda packages
################################################################################

gpuci_logger "Starting conda uploads"

if [[ "$BUILD_LIBRAFT" == "1" && "$UPLOAD_LIBRAFT" == "1" ]]; then
  LIBRAFT_FILES=$(conda build --no-build-id --croot ${CONDA_BLD_DIR} -c ${CONDA_LOCAL_CHANNEL} conda/recipes/libraft --output)
  echo "Upload libraft-headers, libraft-nn, libraft-distance and libraft-tests"
  gpuci_retry anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing --no-progress ${LIBRAFT_FILES}
fi

if [[ "$BUILD_RAFT" == "1" && "$UPLOAD_RAFT" == "1" ]]; then
  RAFT_DASK_FILE=$(conda build --no-build-id --croot ${CONDA_BLD_DIR} -c ${CONDA_LOCAL_CHANNEL} conda/recipes/raft-dask --python=$PYTHON --output)
  PYLIBRAFT_FILE=$(conda build --no-build-id --croot ${CONDA_BLD_DIR} -c ${CONDA_LOCAL_CHANNEL} conda/recipes/pylibraft --python=$PYTHON --output)
  test -e ${RAFT_DASK_FILE}
  echo "Upload raft-dask"
  echo ${RAFT_DASK_FILE}
  gpuci_retry anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing ${RAFT_DASK_FILE} --no-progress

  test -e ${PYLIBRAFT_FILE}
  echo "Upload pylibraft"
  echo ${PYLIBRAFT_FILE}
  gpuci_retry anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing ${PYLIBRAFT_FILE} --no-progress
fi
