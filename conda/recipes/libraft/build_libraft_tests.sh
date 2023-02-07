#!/usr/bin/env bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

./build.sh tests bench --allgpuarch --no-nvtx
cmake --install cpp/build --component testing

BUILD_PATH=$(expand /opt/conda/conda-bld/*libraft-split*/cpp/build/)
TESTS_FILE=libraft.tests.ninja_log

cp ${BUILD_PATH}.ninja_log ${BUILD_PATH}${TESTS_FILE}