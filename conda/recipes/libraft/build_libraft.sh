#!/usr/bin/env bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

./build.sh libraft -v --allgpuarch --compile-lib --build-metrics=compile_lib --incl-cache-stats --no-nvtx
