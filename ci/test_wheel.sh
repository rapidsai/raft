#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_name="${1}"
package_dir="python/${package_name}"
underscore_package_name=$(echo "${package_name}" | tr "-" "_")

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
WHEELHOUSE="${PWD}/dist/"
RAPIDS_PY_WHEEL_NAME="raft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp "${WHEELHOUSE}"
RAPIDS_PY_WHEEL_NAME="raft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python "${WHEELHOUSE}"

python -m pip install "${package_name}-${RAPIDS_PY_CUDA_SUFFIX}[test]>=0.0.0a0" --find-links "${WHEELHOUSE}"

python -m pytest ${package_dir}/${underscore_package_name}/test
