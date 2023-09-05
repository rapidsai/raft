#!/usr/bin/env bash
# Copyright (c) 2023, NVIDIA CORPORATION.

./build.sh bench-ann --cpu-only --no-nvtx --build-metrics=bench_ann_cpu --incl-cache-stats
cmake --install cpp/build --component ann_bench
