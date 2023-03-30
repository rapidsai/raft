#!/usr/bin/env bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

# Just building template so we verify it uses libraft.so and fail if it doesn't build
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS} -ccbin ${CXX}" # Needed for CUDA 12 nvidia channel compilers
./build.sh template
