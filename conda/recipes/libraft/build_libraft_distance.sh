#!/usr/bin/env bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

./build.sh libraft -v --allgpuarch --compile-dist --no-nvtx

BUILD_PATH=/opt/conda/conda-bld/*libraft-split*/cpp/build/
DIST_FILE=libraft.distance.ninja_log

cp ${BUILD_PATH}.ninja_log ${BUILD_PATH}${DIST_FILE}