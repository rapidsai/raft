#!/usr/bin/env bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

# We must install everything (not just the "raft" component) because some
# dependencies like cuCollections and cutlass place their install rules in the
# "all" component.
./build.sh libraft --allgpuarch --no-nvtx
