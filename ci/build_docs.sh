#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

RAPIDS_VERSION="$(rapids-version)"

rapids-dependency-file-generator \
  --output conda \
  --file-key docs \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n docs
conda activate docs

rapids-print-env

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  --channel "${PYTHON_CHANNEL}" \
  "libraft=${RAPIDS_VERSION}" \
  "libraft-headers=${RAPIDS_VERSION}" \
  "pylibraft=${RAPIDS_VERSION}" \
  "raft-dask=${RAPIDS_VERSION}"

export RAPIDS_DOCS_DIR="$(mktemp -d)"

rapids-logger "Build CPP docs"
pushd cpp/doxygen
doxygen Doxyfile
popd

rapids-logger "Build Python docs"
pushd docs
sphinx-build -b dirhtml source _html
mkdir -p "${RAPIDS_DOCS_DIR}/raft/"html
mv _html/* "${RAPIDS_DOCS_DIR}/raft/html"
popd

RAPIDS_VERSION_NUMBER="$(rapids-version-major-minor)" rapids-upload-docs
