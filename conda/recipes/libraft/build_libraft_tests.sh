#!/usr/bin/env bash
# Copyright (c) 2022, NVIDIA CORPORATION.

./build.sh tests bench -v --allgpuarch --sccache --no-nvtx
cmake --install cpp/build --component testing
