#!/usr/bin/env bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

./build.sh libraft --allgpuarch --no-nvtx -n
cmake --install cpp/build --component raft
