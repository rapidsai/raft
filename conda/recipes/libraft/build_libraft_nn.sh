#!/usr/bin/env bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

PARALLEL_LEVEL=8 ./build.sh libraft -v --allgpuarch --compile-nn --no-nvtx
