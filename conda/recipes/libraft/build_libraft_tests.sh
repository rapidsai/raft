#!/usr/bin/env bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

export SCCACHE_RECACHE=1

./build.sh tests bench --allgpuarch --no-nvtx
cmake --install cpp/build --component testing
