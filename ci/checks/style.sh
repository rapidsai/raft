#!/bin/bash
# Copyright (c) 2020-2022, NVIDIA CORPORATION.
#####################
# RAFT Style Tester #
#####################

# Ignore errors and set path
set +e
PATH=/opt/conda/bin:$PATH

# Activate common conda env
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

FORMAT_FILE_URL=https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-22.12/cmake-format-rapids-cmake.json
export RAPIDS_CMAKE_FORMAT_FILE=/tmp/rapids_cmake_ci/cmake-formats-rapids-cmake.json
mkdir -p $(dirname ${RAPIDS_CMAKE_FORMAT_FILE})
wget -O ${RAPIDS_CMAKE_FORMAT_FILE} ${FORMAT_FILE_URL}

# Run pre-commit checks
pre-commit run --hook-stage manual --all-files

exit $RETVAL
