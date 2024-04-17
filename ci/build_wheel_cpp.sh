#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

package_name="libraft"
package_dir="python/libraft"

source rapids-configure-sccache
source rapids-date-string

version=$(rapids-generate-version)
git_commit=$(git rev-parse HEAD)

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# everywhere except in the final wheel name.
PACKAGE_CUDA_SUFFIX="-${RAPIDS_PY_CUDA_SUFFIX}"

# Patch project metadata files to include the CUDA version suffix and version override.
pyproject_file="${package_dir}/pyproject.toml"
version_file="${package_dir}/${package_name}/_version.py"

sed -i "s/name = \"${package_name}\"/name = \"${package_name}${PACKAGE_CUDA_SUFFIX}\"/g" ${pyproject_file}
echo "${version}" > VERSION
sed -i "/^__git_commit__ / s/= .*/= \"${git_commit}\"/g" ${version_file}

# For nightlies we want to ensure that we're pulling in alphas as well. The
# easiest way to do so is to augment the spec with a constraint containing a
# min alpha version that doesn't affect the version bounds but does allow usage
# of alpha versions for that dependency without --pre
alpha_spec=''
if ! rapids-is-release-build; then
    alpha_spec=',>=0.0.0a0'
fi

sed -r -i "s/librmm(.*)\"/librmm${PACKAGE_CUDA_SUFFIX}\1${alpha_spec}\"/g" ${pyproject_file}

cd "${package_dir}"

librmm_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="librmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact rmm 1529 cpp)

python -m pip wheel . -w dist -vvv --no-deps --disable-pip-version-check --find-links "${librmm_wheelhouse}"

mkdir -p final_dist
python -m auditwheel repair -w final_dist dist/*

# TODO: Remove RAPIDS_PY_WHEEL_NAME once gha-tools includes the cuda version in the artifact name.
RAPIDS_PY_WHEEL_NAME="${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 cpp final_dist
