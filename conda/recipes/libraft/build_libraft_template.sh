#!/usr/bin/env bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

# Just building template so we verify it uses libraft.so and fail if it doesn't build
./build.sh template --no-nvtx
