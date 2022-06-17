#!/usr/bin/env bash
# Copyright (c) 2022, NVIDIA CORPORATION.

./build.sh tests bench -v --allgpuarch --no-nvtx
cmake --install cpp/build --component testing
