#!/usr/bin/env bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

./build.sh libraft --allgpuarch --compile-lib --build-metrics=compile_lib_static --incl-cache-stats --no-nvtx --cmake-args=\"-DBUILD_SHARED_LIBS=OFF\"