#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

package_name=$1
package_dir=$2

# Use gha-tools rapids-pip-wheel-version to generate wheel version then
# update the necessary files
RAPIDS_EPOCH_TIMESTAMP=$(date +%s)
versioneer_override="$(rapids-pip-wheel-version ${RAPIDS_EPOCH_TIMESTAMP})"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CTK_VERSION})"

bash ci/release/apply_wheel_modifications.sh ${versioneer_override} "-${RAPIDS_PY_CUDA_SUFFIX}"
echo "The package name and/or version was modified in the package source. The git diff is:"
git diff

cd "${package_dir}"

# sccache configuration
export SCCACHE_S3_KEY_PREFIX=gha-cibw
export SCCACHE_REGION=us-east-2
export SCCACHE_IDLE_TIMEOUT=32768
export SCCACHE_BUCKET=rapids-sccache-east
export SCCACHE_S3_USE_SSL=true

# Set up for pip installation of dependencies from the nightly index
export PIP_EXTRA_INDEX_URL=https://pypi.k8s.rapids.ai/simple

# Hardcode the output dir
python -m pip wheel . -w dist -vvv --no-deps --disable-pip-version-check

# Repair the wheel
cd dist
if [[ "${RAPIDS_ARCH}" == "amd64" ]]; then
    python -m auditwheel repair -w . ${package_name}* --plat manylinux_2_17_x86_64
else
    python -m auditwheel repair -w . ${package_name}* --plat manylinux_2_31_aarch64
fi

# Need to pick the final wheel out from all the dependencies and the
# pre-repaired wheel.
cd ..
mkdir final_dist
mv dist/${package_name}*manylinux* final_dist

# rapids-upload-wheels-to-s3 uses rapids-package-name which implicitly relies
# on this variable being set
export RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}"
rapids-upload-wheels-to-s3 final_dist
