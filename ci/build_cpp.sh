#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

grep -v "conda-forge" /opt/conda/.condarc > /opt/conda/.condarc.tmp
mv /opt/conda/.condarc.tmp /opt/conda/.condarc
echo "Contents of /opt/conda/.condarc:"
cat /opt/conda/.condarc

rapids-print-env

rapids-logger "Begin cpp build"

rapids-conda-retry mambabuild conda/recipes/libraft

rapids-upload-conda-to-s3 cpp
