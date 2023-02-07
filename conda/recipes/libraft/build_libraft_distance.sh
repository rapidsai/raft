#!/usr/bin/env bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

./build.sh libraft -v --allgpuarch --compile-dist --no-nvtx

cp ${BUILD_PATH}.ninja_log ${BUILD_PATH}${DIST_FILE}