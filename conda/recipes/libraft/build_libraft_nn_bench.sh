#!/usr/bin/env bash
# Copyright (c) 2023, NVIDIA CORPORATION.

./build.sh tests bench-ann --allgpuarch --no-nvtx
cmake --install cpp/build --component ann_bench
