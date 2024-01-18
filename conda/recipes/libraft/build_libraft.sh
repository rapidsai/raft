#!/usr/bin/env bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

./build.sh libraft --allgpuarch --compile-lib --build-metrics=compile_lib --incl-cache-stats --no-nvtx
