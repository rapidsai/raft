#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

# Replace conda-forge with nvidia/label/tst.
sed -i "s/conda-forge/nvidia\/label\/tst/g" /opt/conda/.condarc
echo "Contents of /opt/conda/.condarc:"
cat /opt/conda/.condarc

rapids-print-env

rapids-logger "Begin cpp build"

rapids-conda-retry mambabuild conda/recipes/libraft

rapids-upload-conda-to-s3 cpp
