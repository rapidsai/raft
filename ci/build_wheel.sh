#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

package_name=$1
package_dir=$2

# Use gha-tools rapids-pip-wheel-version to generate wheel version then
# update the necessary files
RAPIDS_EPOCH_TIMESTAMP=$(date +%s)
versioneer_override="$(rapids-pip-wheel-version ${RAPIDS_EPOCH_TIMESTAMP})"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

bash ci/release/apply_wheel_modifications.sh ${versioneer_override} "-${RAPIDS_PY_CUDA_SUFFIX}"
echo "The package name and/or version was modified in the package source. The git diff is:"
git diff

cd "${package_dir}"

# sccache configuration
if [ "${CI:-false}" = "false" ]; then
  # Configure sccache for read-only mode since no credentials
  # are available in local builds.
  export SCCACHE_S3_NO_CREDENTIALS=true
  export PARALLEL_LEVEL=${PARALLEL_LEVEL:-$(nproc)}
  export RAPIDS_BUILD_TYPE=${RAPIDS_BUILD_TYPE:-"pull-request"}
fi
export SCCACHE_S3_KEY_PREFIX="libraft-$(arch)"

# Set up for pip installation of dependencies from the nightly index
export PIP_EXTRA_INDEX_URL=https://pypi.k8s.rapids.ai/simple

# Hardcode the output dir
python -m pip wheel . -w dist -vvv --no-deps --disable-pip-version-check

mkdir -p final_dist
python -m auditwheel repair -w final_dist dist/*

if [[ ! -d "/tmp/gha-tools" ]]; then
  git clone https://github.com/divyegala/gha-tools.git -b wheel-local-runs /tmp/gha-tools
fi

RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" /tmp/gha-tools/tools/rapids-upload-wheels-to-s3 final_dist
