#!/usr/bin/env bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

export SCCACHE_RECACHE=0
./build.sh libraft --allgpuarch --compile-nn --no-nvtx
