#!/usr/bin/env bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

./build.sh libraft --allgpuarch --compile-dist --no-nvtx

BUILD_PATH=/opt/conda/conda-bld/*libraft-split*/cpp/build/
DIST_FILE=libraft.distance.ninja_log

FROM_FILE=$(echo ${BUILD_PATH}.ninja_log)
TO_FILE=$(echo ${BUILD_PATH}${DIST_FILE})

cp ${BUILD_PATH}.ninja_log ${BUILD_PATH}${DIST_FILE}
cp $FROM_FILE $TO_FILE