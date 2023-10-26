#!/bin/bash
# Copyright (c) 2020-2023, NVIDIA CORPORATION.
########################
# RAFT Version Updater #
########################

## Usage
# bash update-version.sh <new_version>


# Format is YY.MM.PP - no leading 'v' or trailing 'a'
NEXT_FULL_TAG=$1

# Get current version
CURRENT_TAG=$(git tag --merged HEAD | grep -xE '^v.*' | sort --version-sort | tail -n 1 | tr -d 'v')
CURRENT_MAJOR=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[1]}')
CURRENT_MINOR=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[2]}')
CURRENT_PATCH=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[3]}')
CURRENT_SHORT_TAG=${CURRENT_MAJOR}.${CURRENT_MINOR}

#Get <major>.<minor> for next version
NEXT_MAJOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[2]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}
NEXT_UCX_PY_SHORT_TAG="$(curl -sL https://version.gpuci.io/rapids/${NEXT_SHORT_TAG})"
NEXT_UCX_PY_VERSION="${NEXT_UCX_PY_SHORT_TAG}.*"

# Need to distutils-normalize the original version
NEXT_SHORT_TAG_PEP440=$(python -c "from setuptools.extern import packaging; print(packaging.version.Version('${NEXT_SHORT_TAG}'))")
NEXT_UCX_PY_SHORT_TAG_PEP440=$(python -c "from setuptools.extern import packaging; print(packaging.version.Version('${NEXT_UCX_PY_SHORT_TAG}'))")

echo "Preparing release $CURRENT_TAG => $NEXT_FULL_TAG"

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' $2 && rm -f ${2}.bak
}

sed_runner "s/set(RAPIDS_VERSION .*)/set(RAPIDS_VERSION \"${NEXT_SHORT_TAG}\")/g" cpp/CMakeLists.txt
sed_runner "s/set(RAPIDS_VERSION .*)/set(RAPIDS_VERSION \"${NEXT_SHORT_TAG}\")/g" cpp/template/cmake/thirdparty/fetch_rapids.cmake
sed_runner "s/set(RAFT_VERSION .*)/set(RAFT_VERSION \"${NEXT_FULL_TAG}\")/g" cpp/CMakeLists.txt
sed_runner 's/'"pylibraft_version .*)"'/'"pylibraft_version ${NEXT_FULL_TAG})"'/g' python/pylibraft/CMakeLists.txt
sed_runner 's/'"raft_dask_version .*)"'/'"raft_dask_version ${NEXT_FULL_TAG})"'/g' python/raft-dask/CMakeLists.txt
sed_runner 's/'"branch-.*\/RAPIDS.cmake"'/'"branch-${NEXT_SHORT_TAG}\/RAPIDS.cmake"'/g' fetch_rapids.cmake

# Python __init__.py updates
sed_runner "s/__version__ = .*/__version__ = \"${NEXT_FULL_TAG}\"/g" python/pylibraft/pylibraft/__init__.py
sed_runner "s/__version__ = .*/__version__ = \"${NEXT_FULL_TAG}\"/g" python/raft-dask/raft_dask/__init__.py

# Wheel testing script
sed_runner "s/branch-.*/branch-${NEXT_SHORT_TAG}/g" ci/test_wheel_raft_dask.sh

# Docs update
sed_runner 's/version = .*/version = '"'${NEXT_SHORT_TAG}'"'/g' docs/source/conf.py
sed_runner 's/release = .*/release = '"'${NEXT_FULL_TAG}'"'/g' docs/source/conf.py

DEPENDENCIES=(
  dask-cuda
  pylibraft
  rmm
  # ucx-py is handled separately below
)
for FILE in dependencies.yaml conda/environments/*.yaml; do
  for DEP in "${DEPENDENCIES[@]}"; do
    sed_runner "/-.* ${DEP}==/ s/==.*/==${NEXT_SHORT_TAG_PEP440}\.*/g" ${FILE};
  done
  sed_runner "/-.* ucx-py==/ s/==.*/==${NEXT_UCX_PY_SHORT_TAG_PEP440}\.*/g" ${FILE};
done
for FILE in python/*/pyproject.toml; do
  for DEP in "${DEPENDENCIES[@]}"; do
    sed_runner "/\"${DEP}==/ s/==.*\"/==${NEXT_SHORT_TAG_PEP440}.*\"/g" ${FILE}
  done
  sed_runner "s/^version = .*/version = \"${NEXT_FULL_TAG}\"/g" "${FILE}"
  sed_runner "/\"ucx-py==/ s/==.*\"/==${NEXT_UCX_PY_SHORT_TAG_PEP440}.*\"/g" ${FILE}
done

sed_runner "/^ucx_py_version:$/ {n;s/.*/  - \"${NEXT_UCX_PY_VERSION}\"/}" conda/recipes/raft-dask/conda_build_config.yaml

# Wheel builds install dask-cuda from source, update its branch
for FILE in .github/workflows/*.yaml; do
  sed_runner "s/dask-cuda.git@branch-[^\"\s]\+/dask-cuda.git@branch-${NEXT_SHORT_TAG}/g" ${FILE};
done

for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-workflows/ s/@.*/@branch-${NEXT_SHORT_TAG}/g" "${FILE}"
done
sed_runner "s/RAPIDS_VERSION_NUMBER=\".*/RAPIDS_VERSION_NUMBER=\"${NEXT_SHORT_TAG}\"/g" ci/build_docs.sh

sed_runner "/^PROJECT_NUMBER/ s|\".*\"|\"${NEXT_SHORT_TAG}\"|g" cpp/doxygen/Doxyfile

sed_runner "/^set(RAFT_VERSION/ s|\".*\"|\"${NEXT_SHORT_TAG}\"|g" docs/source/build.md
sed_runner "/GIT_TAG.*branch-/ s|branch-.*|branch-${NEXT_SHORT_TAG}|g" docs/source/build.md
sed_runner "/rapidsai\/raft/ s|branch-[0-9][0-9].[0-9][0-9]|branch-${NEXT_SHORT_TAG}|g" docs/source/developer_guide.md

# .devcontainer files
find .devcontainer/ -type f -name devcontainer.json -print0 | while IFS= read -r -d '' filename; do
    sed_runner "s@rapidsai/devcontainers:[0-9.]*@rapidsai/devcontainers:${NEXT_SHORT_TAG}@g" "${filename}"
    sed_runner "s@rapidsai/devcontainers/features/ucx:[0-9.]*@rapidsai/devcontainers/features/ucx:${NEXT_SHORT_TAG_PEP440}@" "${filename}"
    sed_runner "s@rapidsai/devcontainers/features/rapids-build-utils:[0-9.]*@rapidsai/devcontainers/features/rapids-build-utils:${NEXT_SHORT_TAG_PEP440}@" "${filename}"
done
