#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

if [[ ! -d "/tmp/gha-tools" ]]; then
    git clone https://github.com/msarahan/gha-tools.git -b get-pr-wheel-artifact /tmp/gha-tools
fi
export PATH="/tmp/gha-tools/tools:${PATH}"

source rapids-configure-sccache
source rapids-date-string

version=$(rapids-generate-version)
git_commit=$(git rev-parse HEAD)

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="libraft_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp /tmp/libraft_dist

# This is the version of the suffix with a preceding hyphen. It's used
# everywhere except in the final wheel name.
PACKAGE_CUDA_SUFFIX="-${RAPIDS_PY_CUDA_SUFFIX}"

echo "${version}" > VERSION
# For nightlies we want to ensure that we're pulling in alphas as well. The
# easiest way to do so is to augment the spec with a constraint containing a
# min alpha version that doesn't affect the version bounds but does allow usage
# of alpha versions for that dependency without --pre
alpha_spec=''
if ! rapids-is-release-build; then
    alpha_spec=',>=0.0.0a0'
fi


###############################################
# Build pylibraft

librmm_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="librmm_${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact rmm 1512 cpp)

package_name="pylibraft"
package_dir="python/pylibraft"

pyproject_file="${package_dir}/pyproject.toml"
version_file="${package_dir}/${package_name}/_version.py"

sed -i "s/name = \"${package_name}\"/name = \"${package_name}${PACKAGE_CUDA_SUFFIX}\"/g" ${pyproject_file}
sed -i "/^__git_commit__ / s/= .*/= \"${git_commit}\"/g" ${version_file}

sed -r -i "s/rmm(.*)\"/rmm${PACKAGE_CUDA_SUFFIX}\1${alpha_spec}\"/g" ${pyproject_file}
sed -r -i "s/libraft==(.*)\"/libraft${PACKAGE_CUDA_SUFFIX}==\1${alpha_spec}\"/g" ${pyproject_file}
if [[ $PACKAGE_CUDA_SUFFIX == "-cu12" ]]; then
    sed -i "s/cuda-python[<=>\.,0-9a]*/cuda-python>=12.0,<13.0a0/g" ${pyproject_file}
    sed -i "s/cupy-cuda11x/cupy-cuda12x/g" ${pyproject_file}
fi

pushd "${package_dir}"

PIP_FIND_LINKS="/tmp/libraft_dist ${librmm_wheelhouse}" python -m pip wheel . -w pylibraft_dist -vvv --no-deps --disable-pip-version-check

mkdir -p pylibraft_final_dist
python -m auditwheel repair -w pylibraft_final_dist --exclude libraft.so pylibraft_dist/*

RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 python pylibraft_final_dist

popd


################################################
# Build raft-dask

package_name="raft-dask"
package_dir="python/raft-dask"
underscore_package_name=$(echo "${package_name}" | tr "-" "_")

pyproject_file="${package_dir}/pyproject.toml"
version_file="${package_dir}/${underscore_package_name}/_version.py"

sed -i "s/name = \"${package_name}\"/name = \"${package_name}${PACKAGE_CUDA_SUFFIX}\"/g" ${pyproject_file}
sed -i "/^__git_commit__ / s/= .*/= \"${git_commit}\"/g" ${version_file}

sed -r -i "s/libraft(.*)\"/libraft${PACKAGE_CUDA_SUFFIX}\1${alpha_spec}\"/g" ${pyproject_file}
sed -r -i "s/pylibraft==(.*)\"/pylibraft${PACKAGE_CUDA_SUFFIX}==\1${alpha_spec}\"/g" ${pyproject_file}
sed -r -i "s/ucx-py==(.*)\"/ucx-py${PACKAGE_CUDA_SUFFIX}==\1${alpha_spec}\"/g" ${pyproject_file}
sed -r -i "s/rapids-dask-dependency==(.*)\"/rapids-dask-dependency==\1${alpha_spec}\"/g" ${pyproject_file}
sed -r -i "s/dask-cuda==(.*)\"/dask-cuda==\1${alpha_spec}\"/g" ${pyproject_file}

pushd "${package_dir}"

PIP_FIND_LINKS="../pylibraft/pylibraft_dist /tmp/libraft_dist" python -m pip wheel . -w raft_dask_dist -vvv --no-deps --disable-pip-version-check

mkdir -p raft_dask_final_dist
python -m auditwheel repair -w raft_dask_final_dist --exclude libraft.so raft_dask_dist/*

RAPIDS_PY_WHEEL_NAME="${underscore_package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 raft_dask_final_dist

popd
