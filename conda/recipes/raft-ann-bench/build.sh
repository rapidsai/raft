#!/usr/bin/env bash
# Copyright (c) 2023, NVIDIA CORPORATION.

./build.sh bench-ann -v --allgpuarch --no-nvtx --build-metrics=bench_ann --incl-cache-stats
cmake --install cpp/build --component ann_bench
