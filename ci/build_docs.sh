#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-dependency-file-generator \
  --output conda \
  --file_key docs \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n docs
conda activate docs

rapids-print-env

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)
VERSION_NUMBER=$(rapids-get-rapids-version-from-git)

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  --channel "${PYTHON_CHANNEL}" \
  libraft-distance \
  libraft-headers \
  libraft-nn \
  pylibraft \
  raft-dask


rapids-logger "Build Doxygen docs"
pushd cpp/doxygen
doxygen Doxyfile
popd

rapids-logger "Build Sphinx docs"
pushd docs
sphinx-build -b dirhtml source _html
sphinx-build -b text source _text
popd


if [[ ${RAPIDS_BUILD_TYPE} == "branch" ]]; then
  rapids-logger "Upload Docs to S3"
  aws s3 sync --no-progress --delete docs/_html "s3://rapidsai-docs/raft/${VERSION_NUMBER}/html"
  aws s3 sync --no-progress --delete docs/_text "s3://rapidsai-docs/raft/${VERSION_NUMBER}/txt"
fi
