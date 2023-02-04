#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin cpp build"

rapids-mamba-retry mambabuild conda/recipes/libraft

rapids-upload-conda-to-s3 cpp


rapids-logger "ls conda-bld"

ls -al /opt/conda/conda-bld/

rapids-logger "ls conda-bld/*libraft*/"

ls -al /opt/conda/conda-bld/*libraft*/**/*

# stop the CI pipeline here to preserve resources
exit 1
