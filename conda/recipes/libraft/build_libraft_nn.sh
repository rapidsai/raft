#!/usr/bin/env bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

./build.sh libraft -v --allgpuarch --compile-nn --no-nvtx

BUILD_PATH=$(expand /opt/conda/conda-bld/*libraft-split*/cpp/build/)
NN_FILE=libraft.nn.ninja_log

cp ${BUILD_PATH}.ninja_log ${BUILD_PATH}${NN_FILE}