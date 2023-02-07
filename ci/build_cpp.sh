#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja
BUILD_PATH=/opt/conda/conda-bld/*libraft-split*/cpp/build/
DIST_FILE=libraft.distance.ninja_log
NN_FILE=libraft.nn.ninja_log
TESTS_FILE=libraft.tests.ninja_log

rapids-print-env

rapids-logger "Begin cpp build"

rapids-mamba-retry mambabuild conda/recipes/libraft

rapids-upload-conda-to-s3 cpp


rapids-logger "ls conda-bld"

ls -al /opt/conda/conda-bld/

rapids-logger "ls conda-bld/*libraft*/"

UPLOAD_NAME=cpp_cuda${RAPIDS_CUDA_VERSION%%.*}_$(arch)
DIST_UPLOAD_NAME=${UPLOAD_NAME}.${DIST_FILE}
NN_UPLOAD_NAME=${UPLOAD_NAME}.${NN_FILE}
TESTS_UPLOAD_NAME=${UPLOAD_NAME}.${TESTS_FILE}

rapids-upload-to-s3 "${DIST_UPLOAD_NAME}" "${BUILD_PATH}${DIST_FILE}"
rapids-upload-to-s3 "${NN_UPLOAD_NAME}" "${BUILD_PATH}${NN_FILE}"
rapids-upload-to-s3 "${TESTS_UPLOAD_NAME}" "${BUILD_PATH}${TESTS_FILE}"

