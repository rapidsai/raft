#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-dependency-file-generator \
  --output conda \
  --file_key docs \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-dependency-file-generator \
  --output requirements \
  --file_key docs \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee requirements.txt
  
rapids-mamba-retry env create --force -f env.yaml -n docs
conda activate docs
pip install -r requirements.txt

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


# Build CPP docs
rapids-logger "Build CPP docs"
#pushd docs
#sphinx-build -b dirhtml source build -W
#popd
pushd ../
build.sh docs -v -n
popd


if [[ ${RAPIDS_BUILD_TYPE} == "branch" ]]; then
  rapids-logger "Upload Docs to S3"
  aws s3 sync --delete docs/build "s3://rapidsai-docs/raft/${VERSION_NUMBER}/build"
fi
