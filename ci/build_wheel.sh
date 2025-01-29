#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_name=$1
package_dir=$2
package_type=$3
underscore_package_name=$(echo "${package_name}" | tr "-" "_")

# Clear out system ucx files to ensure that we're getting ucx from the wheel.
rm -rf /usr/lib64/ucx
rm -rf /usr/lib64/libuc*

source rapids-configure-sccache
source rapids-date-string

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

rapids-generate-version > ./VERSION

cd "${package_dir}"

EXCLUDE_ARGS=(
  --exclude "libcublas.so.*"
  --exclude "libcublasLt.so.*"
  --exclude "libcurand.so.*"
  --exclude "libcusolver.so.*"
  --exclude "libcusparse.so.*"
  --exclude "libnvJitLink.so.*"
  --exclude "libucp.so.*"
)

if [[ ${package_name} != "libraft" ]]; then
    EXCLUDE_ARGS+=(
      --exclude "libraft.so"
    )
fi


# TODO: remove me once gha-tools is released
if ! command -v "rapids-pip-retry" &> /dev/null
then
    git clone --branch rapids-pip-retry https://github.com/gforsyth/gha-tools.git

    export PATH="$PWD/gha-tools/tools":$PATH
fi

sccache --zero-stats

rapids-logger "Building '${package_name}' wheel"

# I am so sorry:
for i in {1..20}; do
  rapids-pip-retry wheel \
      --force-reinstall \
      -w dist \
      -v \
      --no-deps \
      --disable-pip-version-check \
      .
done

sccache --show-adv-stats

mkdir -p final_dist
python -m auditwheel repair -w final_dist "${EXCLUDE_ARGS[@]}" dist/*

RAPIDS_PY_WHEEL_NAME="${underscore_package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 ${package_type} final_dist
