#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

source rapids-init-pip

package_name=$1
package_dir=$2

# Clear out system ucx files to ensure that we're getting ucx from the wheel.
rm -rf /usr/lib64/ucx
rm -rf /usr/lib64/libuc*

source rapids-configure-sccache
source rapids-date-string

rapids-generate-version > ./VERSION

cd "${package_dir}"

EXCLUDE_ARGS=(
  --exclude "libcublas.so.*"
  --exclude "libcublasLt.so.*"
  --exclude "libcurand.so.*"
  --exclude "libcusolver.so.*"
  --exclude "libcusparse.so.*"
  --exclude "libnvJitLink.so.*"
  --exclude "librapids_logger.so"
  --exclude "librmm.so"
  --exclude "libucp.so.*"
)

if [[ ${package_name} != "libraft" ]]; then
    EXCLUDE_ARGS+=(
      --exclude "libraft.so"
    )
fi

RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
if [[ ${RAPIDS_CUDA_MAJOR} == "12" ]]; then
    EXCLUDE_ARGS+=(
      --exclude "libnccl.so.*"
    )
    export SKBUILD_CMAKE_ARGS="-DUSE_NCCL_RUNTIME_WHEEL=ON"
fi

sccache --zero-stats

rapids-logger "Building '${package_name}' wheel"

rapids-pip-retry wheel \
    -w dist \
    -v \
    --no-deps \
    --disable-pip-version-check \
    .

sccache --show-adv-stats

# repair wheels and write to the location that artifact-uploading code expects to find them
python -m auditwheel repair -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" "${EXCLUDE_ARGS[@]}" dist/*
