#!/usr/bin/env bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS} -ccbin ${CXX}" # Needed for CUDA 12 nvidia channel compilers
./build.sh libraft --allgpuarch --compile-lib --build-metrics=compile_lib --incl-cache-stats --no-nvtx
