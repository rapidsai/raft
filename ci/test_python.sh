#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

set -euo pipefail

# Support invoking test_python.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

. /opt/conda/etc/profile.d/conda.sh

RAPIDS_VERSION_MAJOR_MINOR="$(rapids-version-major-minor)"

rapids-logger "Generate Python testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

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
  "libraft=${RAPIDS_VERSION_MAJOR_MINOR}" \
  "libraft-headers=${RAPIDS_VERSION_MAJOR_MINOR}" \
  "pylibraft=${RAPIDS_VERSION_MAJOR_MINOR}" \
  "raft-dask=${RAPIDS_VERSION_MAJOR_MINOR}"

rapids-logger "Check GPU usage"
nvidia-smi

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest pylibraft"
./ci/run_pylibraft_pytests.sh \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-pylibraft.xml" \
  --cov-config=../.coveragerc \
  --cov=pylibraft \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/pylibraft-coverage.xml" \
  --cov-report=term

rapids-logger "pytest raft-dask"
./ci/run_raft_dask_pytests.sh \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-raft-dask.xml" \
  --cov-config=../.coveragerc \
  --cov=raft_dask \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/raft-dask-coverage.xml" \
  --cov-report=term

rapids-logger "pytest raft-dask (ucx-py only)"
./ci/run_raft_dask_pytests.sh \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-raft-dask-ucx.xml" \
  --cov-config=../.coveragerc \
  --cov=raft_dask \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/raft-dask-ucx-coverage.xml" \
  --cov-report=term \
  --run_ucx

rapids-logger "pytest raft-dask (ucxx only)"
./ci/run_raft_dask_pytests.sh \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-raft-dask-ucxx.xml" \
  --cov-config=../.coveragerc \
  --cov=raft_dask \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/raft-dask-ucxx-coverage.xml" \
  --cov-report=term \
  --run_ucxx

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
