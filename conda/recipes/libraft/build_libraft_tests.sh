#!/usr/bin/env bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

./build.sh tests bench-prims --allgpuarch --no-nvtx --build-metrics=tests_bench_prims --incl-cache-stats
cmake --install cpp/build --component testing
