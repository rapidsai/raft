#!/usr/bin/env bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

./build.sh tests bench --allgpuarch --no-nvtx
cmake --install cpp/build --component testing

BUILD_PATH=/opt/conda/conda-bld/*libraft-split*/cpp/build/
TESTS_FILE=libraft.tests.ninja_log

FROM_FILE=$(echo {BUILD_PATH}.ninja_log)
TO_FILE=$(echo ${BUILD_PATH}${TESTS_FILE})

cp $FROM_FILE $TO_FILE