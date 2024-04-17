#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

source rapids-configure-sccache
source rapids-date-string

version=$(rapids-generate-version)
git_commit=$(git rev-parse HEAD)

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

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

# TODO: Remove RAPIDS_PY_WHEEL_NAME once gha-tools includes the cuda version in the artifact name.
# TODO: Remove the final argument once rapids-download-wheels-from-s3 is updated to construct the directory.
CPP_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp /tmp/libraft_dist)
librmm_wheelhouse=$(RAPIDS_PY_WHEEL_NAME="${RAPIDS_PY_CUDA_SUFFIX}" rapids-get-pr-wheel-artifact rmm 1529 cpp)
PYTHON_WHEELHOUSE="${PWD}/dist/"
PYTHON_AUDITED_WHEELHOUSE="${PWD}/final_dist/"
WHEELHOUSES=("${PYTHON_WHEELHOUSE}" "${CPP_WHEELHOUSE}" "${librmm_wheelhouse}")
mkdir -p "${PYTHON_AUDITED_WHEELHOUSE}"

FIND_LINKS=""
# Iterate over the array
for wheelhouse in "${WHEELHOUSES[@]}"; do
    FIND_LINKS+="--find-links ${wheelhouse} "
done
              

build_wheel () {
    local package_name="${1}"
    local underscore_package_name=$(echo "${package_name}" | tr "-" "_")

    local package_dir="python/${package_name}"
    local pyproject_file="${package_dir}/pyproject.toml"
    local version_file="${package_dir}/${underscore_package_name}/_version.py"

    sed -i "s/name = \"${package_name}\"/name = \"${package_name}${PACKAGE_CUDA_SUFFIX}\"/g" ${pyproject_file}
    sed -i "/^__git_commit__ / s/= .*/= \"${git_commit}\"/g" ${version_file}

    for dep in rmm libraft pylibraft ucx-py; do
        sed -r -i "s/${dep}==(.*)\"/${dep}${PACKAGE_CUDA_SUFFIX}==\1${alpha_spec}\"/g" ${pyproject_file}
    done

    # dask-cuda & rapids-dask-dependency don't get a suffix, but they do get an alpha spec.
    for dep in dask-cuda rapids-dask-dependency; do
        sed -r -i "s/${dep}==(.*)\"/${dep}==\1${alpha_spec}\"/g" ${pyproject_file}
    done

    if [[ $PACKAGE_CUDA_SUFFIX == "-cu12" ]]; then
        sed -i "s/cuda-python[<=>\.,0-9a]*/cuda-python>=12.0,<13.0a0/g" ${pyproject_file}
        sed -i "s/cupy-cuda11x/cupy-cuda12x/g" ${pyproject_file}
    fi

    pushd "${package_dir}"

    python -m pip wheel . -w "${PYTHON_WHEELHOUSE}" -vvv --no-deps --disable-pip-version-check ${FIND_LINKS}
    popd
}

build_wheel pylibraft
build_wheel raft-dask

python -m auditwheel repair -w "${PYTHON_AUDITED_WHEELHOUSE}" --exclude libraft.so "${PYTHON_WHEELHOUSE}"/*
RAPIDS_PY_WHEEL_NAME="${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 python "${PYTHON_AUDITED_WHEELHOUSE}"
