#!/bin/bash
# Copyright (c) 2020-2024, NVIDIA CORPORATION.
########################
# RAFT Version Updater #
########################

## Usage
# bash update-version.sh <new_version>


# Format is YY.MM.PP - no leading 'v' or trailing 'a'
NEXT_FULL_TAG=$1

# Get current version
CURRENT_TAG=$(git tag --merged HEAD | grep -xE '^v.*' | sort --version-sort | tail -n 1 | tr -d 'v')
CURRENT_MAJOR=$(echo "$CURRENT_TAG" | awk '{split($0, a, "."); print a[1]}')
CURRENT_MINOR=$(echo "$CURRENT_TAG" | awk '{split($0, a, "."); print a[2]}')
CURRENT_SHORT_TAG=${CURRENT_MAJOR}.${CURRENT_MINOR}

# Get <major>.<minor> for next version
NEXT_MAJOR=$(echo "$NEXT_FULL_TAG" | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo "$NEXT_FULL_TAG" | awk '{split($0, a, "."); print a[2]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}
NEXT_UCXX_SHORT_TAG="$(curl -sL https://version.gpuci.io/rapids/"${NEXT_SHORT_TAG}")"

# Need to distutils-normalize the original version
NEXT_SHORT_TAG_PEP440=$(python -c "from packaging.version import Version; print(Version('${NEXT_SHORT_TAG}'))")
NEXT_UCXX_SHORT_TAG_PEP440=$(python -c "from packaging.version import Version; print(Version('${NEXT_UCXX_SHORT_TAG}'))")

echo "Preparing release $CURRENT_TAG => $NEXT_FULL_TAG"

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' "$2" && rm -f "${2}".bak
}

sed_runner 's/'"find_and_configure_ucxx(VERSION .*"'/'"find_and_configure_ucxx(VERSION  ${NEXT_UCXX_SHORT_TAG_PEP440}"'/g' python/raft-dask/cmake/thirdparty/get_ucxx.cmake
sed_runner 's/'"branch-.*"'/'"branch-${NEXT_UCXX_SHORT_TAG_PEP440}"'/g' python/raft-dask/cmake/thirdparty/get_ucxx.cmake

# Centralized version file update
echo "${NEXT_FULL_TAG}" > VERSION

DEPENDENCIES=(
  dask-cuda
  libraft
  librmm
  pylibraft
  rmm
  rapids-dask-dependency
  libraft-headers
  raft-dask
  libraft-tests
)
UCXX_DEPENDENCIES=(
  libucxx
  distributed-ucxx
)
for FILE in dependencies.yaml conda/environments/*.yaml; do
  for DEP in "${DEPENDENCIES[@]}"; do
    sed_runner "/-.* ${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}\(\[.*\]\)\{0,1\}==/ s/==.*/==${NEXT_SHORT_TAG_PEP440}.*,>=0.0.0a0/g" "${FILE}"
  done
  for DEP in "${UCXX_DEPENDENCIES[@]}"; do
    sed_runner "/-.* ${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}==/ s/==.*/==${NEXT_UCXX_SHORT_TAG_PEP440}.*,>=0.0.0a0/g" "${FILE}"
  done
done
for FILE in python/*/pyproject.toml; do
  for DEP in "${DEPENDENCIES[@]}"; do
    sed_runner "/\"${DEP}==/ s/==.*\"/==${NEXT_SHORT_TAG_PEP440}.*,>=0.0.0a0\"/g" "${FILE}"
  done
  for DEP in "${UCXX_DEPENDENCIES[@]}"; do
    sed_runner "/\"${DEP}==/ s/==.*\"/==${NEXT_UCXX_SHORT_TAG_PEP440}.*,>=0.0.0a0\"/g" "${FILE}"
  done
done

# RAPIDS UCX version
for FILE in conda/recipes/*/conda_build_config.yaml; do
  sed_runner "/^ucxx_version:$/ {n;s/.*/  - \"${NEXT_UCXX_SHORT_TAG_PEP440}.*\"/}" "${FILE}"
done

for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-workflows/ s/@.*/@branch-${NEXT_SHORT_TAG}/g" "${FILE}"
  sed_runner "s/:[0-9]*\\.[0-9]*-/:${NEXT_SHORT_TAG}-/g" "${FILE}"
done

sed_runner "/^set(RAFT_VERSION/ s|\".*\"|\"${NEXT_SHORT_TAG}\"|g" docs/source/build.md
sed_runner "s|branch-[0-9][0-9].[0-9][0-9]|branch-${NEXT_SHORT_TAG}|g" docs/source/build.md
sed_runner "/rapidsai\/raft/ s|branch-[0-9][0-9].[0-9][0-9]|branch-${NEXT_SHORT_TAG}|g" docs/source/developer_guide.md

sed_runner "s|branch-[0-9][0-9].[0-9][0-9]|branch-${NEXT_SHORT_TAG}|g" README.md

# .devcontainer files
find .devcontainer/ -type f -name devcontainer.json -print0 | while IFS= read -r -d '' filename; do
    sed_runner "s@rapidsai/devcontainers:[0-9.]*@rapidsai/devcontainers:${NEXT_SHORT_TAG}@g" "${filename}"
    sed_runner "s@rapidsai/devcontainers/features/ucx:[0-9.]*@rapidsai/devcontainers/features/ucx:${NEXT_SHORT_TAG_PEP440}@" "${filename}"
    sed_runner "s@rapidsai/devcontainers/features/cuda:[0-9.]*@rapidsai/devcontainers/features/cuda:${NEXT_SHORT_TAG_PEP440}@" "${filename}"
    sed_runner "s@rapidsai/devcontainers/features/rapids-build-utils:[0-9.]*@rapidsai/devcontainers/features/rapids-build-utils:${NEXT_SHORT_TAG_PEP440}@" "${filename}"
    sed_runner "s@rapids-\${localWorkspaceFolderBasename}-[0-9.]*@rapids-\${localWorkspaceFolderBasename}-${NEXT_SHORT_TAG}@g" "${filename}"
done
