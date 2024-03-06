#!/usr/bin/env bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

./build.sh tests bench-prims --allgpuarch --compile-lib --no-nvtx --build-metrics=tests_bench_prims --incl-cache-stats
cmake --install cpp/build --component testing
