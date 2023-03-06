#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Generate Python testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file_key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"

rapids-print-env

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  --channel "${PYTHON_CHANNEL}" \
  libraft-distance libraft-headers pylibraft raft-dask

rapids-logger "Check GPU usage"
nvidia-smi

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest pylibraft"
pushd python/pylibraft/pylibraft
pytest \
  --cache-clear \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-pylibraft.xml" \
  --cov-config=../.coveragerc \
  --cov=pylibraft \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/pylibraft-coverage.xml" \
  --cov-report=term \
  test
popd

rapids-logger "pytest raft-dask"
pushd python/raft-dask/raft_dask
pytest \
  --cache-clear \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-raft-dask.xml" \
  --cov-config=../.coveragerc \
  --cov=raft_dask \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/raft-dask-coverage.xml" \
  --cov-report=term \
  test
popd

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
