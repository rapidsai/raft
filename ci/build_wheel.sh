#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_name=$1
package_dir=$2
underscore_package_name=$(echo "${package_name}" | tr "-" "_")

# Clear out system ucx files to ensure that we're getting ucx from the wheel.
rm -rf /usr/lib64/ucx
rm -rf /usr/lib64/libuc*

source rapids-configure-sccache
source rapids-date-string

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

rapids-generate-version > VERSION

cd "${package_dir}"

case "${RAPIDS_CUDA_VERSION}" in
  12.*)
    EXCLUDE_ARGS=(
      --exclude "libcublas.so.12"
      --exclude "libcublasLt.so.12"
      --exclude "libcurand.so.10"
      --exclude "libcusolver.so.11"
      --exclude "libcusparse.so.12"
      --exclude "libnvJitLink.so.12"
      --exclude "libucp.so.0"
    )
  ;;
  11.*)
    EXCLUDE_ARGS=(
      --exclude "libucp.so.0"
    )
  ;;
esac

LIBRMM_CHANNEL=$(
  RAPIDS_PY_WHEEL_NAME=rmm_${RAPIDS_PY_CUDA_SUFFIX} rapids-get-pr-wheel-artifact rmm 1678 cpp
)
RMM_CHANNEL=$(
  RAPIDS_PY_WHEEL_NAME=rmm_${RAPIDS_PY_CUDA_SUFFIX}_py${RAPIDS_PY_VERSION//./} rapids-get-pr-wheel-artifact rmm 1678 python
)

echo "rmm-${RAPIDS_PY_CUDA_SUFFIX} @ $(echo ${RMM_CHANNEL}/rmm*.whl)" > /tmp/constraints.txt
echo "librmm-${RAPIDS_PY_CUDA_SUFFIX} @ $(echo ${LIBRMM_CHANNEL}/librmm*.whl)" >> /tmp/constraints.txt
echo "" >> /tmp/constraints.txt
export PIP_CONSTRAINT=/tmp/constraints.txt

# Hardcode the output dir
python -m pip wheel . -w dist -v --no-deps --disable-pip-version-check

mkdir -p final_dist
python -m auditwheel repair -w final_dist "${EXCLUDE_ARGS[@]}" dist/*

RAPIDS_PY_WHEEL_NAME="${underscore_package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 final_dist
