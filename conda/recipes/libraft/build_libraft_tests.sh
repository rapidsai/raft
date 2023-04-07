#!/usr/bin/env bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

./build.sh tests bench-prims bench-ann --allgpuarch --no-nvtx
cmake --install cpp/build --component testing
cmake --install cpp/build --component bench-prims
