#!/usr/bin/env bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

PARALLEL_LEVEL=8 ./build.sh tests bench -v --allgpuarch --no-nvtx
cmake --install cpp/build --component testing
