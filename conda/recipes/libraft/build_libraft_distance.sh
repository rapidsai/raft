#!/usr/bin/env bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

./build.sh libraft -v --allgpuarch --compile-dist --no-nvtx

cp /opt/conda/conda-bld/*libraft-split*/cpp/build/.ninja_log /opt/conda/conda-bld/*libraft-split*/cpp/build/libraft.distance.ninja_log