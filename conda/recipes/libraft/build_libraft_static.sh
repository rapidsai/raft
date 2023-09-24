#!/usr/bin/env bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

./build.sh libraft --allgpuarch --compile-static-lib --build-metrics=compile_lib_static --incl-cache-stats --no-nvtx -n
cmake --install cpp/build --component compiled-static
