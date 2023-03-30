#!/usr/bin/env bash
# Copyright (c) 2023, NVIDIA CORPORATION.

export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS} -ccbin ${CXX}" # Needed for CUDA 12 nvidia channel compilers
./build.sh tests bench-ann --allgpuarch --no-nvtx
cmake --install cpp/build --component ann_bench
