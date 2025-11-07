#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
########################
# RAFT Version Updater #
########################

## Usage
# NOTE: This script must be run from the repository root, not from the ci/release/ directory
# Primary interface:   bash ci/release/update-version.sh <new_version> [--run-context=main|release]
# Fallback interface:  [RAPIDS_RUN_CONTEXT=main|release] bash ci/release/update-version.sh <new_version>
# CLI arguments take precedence over environment variables
# Defaults to main when no run-context is specified


# Parse command line arguments
CLI_RUN_CONTEXT=""
VERSION_ARG=""

for arg in "$@"; do
    case ${arg} in
        --run-context=*)
            CLI_RUN_CONTEXT="${arg#*=}"
            shift
            ;;
        *)
            if [[ -z "${VERSION_ARG}" ]]; then
                VERSION_ARG="${arg}"
            fi
            ;;
    esac
done

# Format is YY.MM.PP - no leading 'v' or trailing 'a'
NEXT_FULL_TAG="${VERSION_ARG}"

# Determine RUN_CONTEXT with CLI precedence over environment variable, defaulting to main
if [[ -n "${CLI_RUN_CONTEXT}" ]]; then
    RUN_CONTEXT="${CLI_RUN_CONTEXT}"
    echo "Using run-context from CLI: ${RUN_CONTEXT}"
elif [[ -n "${RAPIDS_RUN_CONTEXT}" ]]; then
    RUN_CONTEXT="${RAPIDS_RUN_CONTEXT}"
    echo "Using run-context from environment: ${RUN_CONTEXT}"
else
    RUN_CONTEXT="main"
    echo "No run-context provided, defaulting to: ${RUN_CONTEXT}"
fi

# Validate RUN_CONTEXT value
if [[ "${RUN_CONTEXT}" != "main" && "${RUN_CONTEXT}" != "release" ]]; then
    echo "Error: Invalid run-context value '${RUN_CONTEXT}'"
    echo "Valid values: main, release"
    exit 1
fi

# Validate version argument
if [[ -z "${NEXT_FULL_TAG}" ]]; then
    echo "Error: Version argument is required"
    echo "Usage: ${0} <new_version> [--run-context=<context>]"
    echo "   or: [RAPIDS_RUN_CONTEXT=<context>] ${0} <new_version>"
    echo "Note: Defaults to main when run-context is not specified"
    exit 1
fi

# Get current version
CURRENT_TAG=$(git tag --merged HEAD | grep -xE '^v.*' | sort --version-sort | tail -n 1 | tr -d 'v')

# Get <major>.<minor> for next version
NEXT_MAJOR=$(echo "${NEXT_FULL_TAG}" | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo "${NEXT_FULL_TAG}" | awk '{split($0, a, "."); print a[2]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}
NEXT_UCXX_SHORT_TAG="$(curl -sL https://version.gpuci.io/rapids/"${NEXT_SHORT_TAG}")"

# Need to distutils-normalize the original version
NEXT_SHORT_TAG_PEP440=$(python -c "from packaging.version import Version; print(Version('${NEXT_SHORT_TAG}'))")
NEXT_UCXX_SHORT_TAG_PEP440=$(python -c "from packaging.version import Version; print(Version('${NEXT_UCXX_SHORT_TAG}'))")

# Set branch references based on RUN_CONTEXT
if [[ "${RUN_CONTEXT}" == "main" ]]; then
    RAPIDS_BRANCH_NAME="main"
    echo "Preparing development branch update ${CURRENT_TAG} => ${NEXT_FULL_TAG} (targeting main branch)"
elif [[ "${RUN_CONTEXT}" == "release" ]]; then
    RAPIDS_BRANCH_NAME="release/${NEXT_SHORT_TAG}"
    echo "Preparing release branch update ${CURRENT_TAG} => ${NEXT_FULL_TAG} (targeting release/${NEXT_SHORT_TAG} branch)"
fi

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"${1}"'' "${2}" && rm -f "${2}".bak
}

# Update UCXX references this does NOT use context-aware branch logic
sed_runner 's/'"find_and_configure_ucxx(VERSION .*"'/'"find_and_configure_ucxx(VERSION  ${NEXT_UCXX_SHORT_TAG_PEP440}"'/g' python/raft-dask/cmake/thirdparty/get_ucxx.cmake
sed_runner 's/'"branch-.*"'/'"branch-${NEXT_UCXX_SHORT_TAG_PEP440}"'/g' python/raft-dask/cmake/thirdparty/get_ucxx.cmake

# Centralized version file update
echo "${NEXT_FULL_TAG}" > VERSION
echo "${RAPIDS_BRANCH_NAME}" > RAPIDS_BRANCH

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
  sed_runner "/^ucxx_version:\$/ {n;s|.*|  - \"${NEXT_UCXX_SHORT_TAG_PEP440}\\.*\"|;}" "${FILE}"
done

# CI files - context-aware branch references
for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-workflows/ s|@.*|@${RAPIDS_BRANCH_NAME}|g" "${FILE}"
  sed_runner "s|:[0-9]*\\.[0-9]*-|:${NEXT_SHORT_TAG}-|g" "${FILE}"
done

# Documentation references - context-aware
sed_runner "/^set(RAFT_VERSION/ s|\".*\"|\"${NEXT_SHORT_TAG}\"|g" docs/source/build.md

if [[ "${RUN_CONTEXT}" == "main" ]]; then
    # In main context, use main branch for documentation links
    sed_runner "s|branch-[0-9][0-9]\\.[0-9][0-9]|main|g" docs/source/build.md
    sed_runner "/rapidsai\\/raft/ s|branch-[0-9][0-9]\\.[0-9][0-9]|main|g" docs/source/developer_guide.md
    sed_runner "s|branch-[0-9][0-9]\\.[0-9][0-9]|main|g" README.md
    # Update CI script URLs to use NEW release/ paradigm
    sed_runner "s|/rapidsai/rapids-cmake/branch-[^/]*/|/rapidsai/rapids-cmake/release/${NEXT_SHORT_TAG}/|g" ci/check_style.sh
    sed_runner "s|/rapidsai/rapids-cmake/main/|/rapidsai/rapids-cmake/release/${NEXT_SHORT_TAG}/|g" ci/check_style.sh
elif [[ "${RUN_CONTEXT}" == "release" ]]; then
    # In release context, use release branch for documentation links
    sed_runner "s|main|branch-${NEXT_SHORT_TAG}|g" docs/source/build.md
    sed_runner "s|branch-[0-9][0-9]\\.[0-9][0-9]|branch-${NEXT_SHORT_TAG}|g" docs/source/build.md
    sed_runner "s|main|branch-${NEXT_SHORT_TAG}|g" docs/source/developer_guide.md
    sed_runner "/rapidsai\\/raft/ s|branch-[0-9][0-9]\\.[0-9][0-9]|branch-${NEXT_SHORT_TAG}|g" docs/source/developer_guide.md
    sed_runner "s|main|branch-${NEXT_SHORT_TAG}|g" README.md
    sed_runner "s|branch-[0-9][0-9]\\.[0-9][0-9]|branch-${NEXT_SHORT_TAG}|g" README.md
    # Update CI script URLs to use NEW release/ paradigm
    sed_runner "s|/rapidsai/rapids-cmake/main/|/rapidsai/rapids-cmake/release/${NEXT_SHORT_TAG}/|g" ci/check_style.sh
fi

# .devcontainer files
find .devcontainer/ -type f -name devcontainer.json -print0 | while IFS= read -r -d '' filename; do
    sed_runner "s@rapidsai/devcontainers:[0-9.]*@rapidsai/devcontainers:${NEXT_SHORT_TAG}@g" "${filename}"
    sed_runner "s@rapidsai/devcontainers/features/ucx:[0-9.]*@rapidsai/devcontainers/features/ucx:${NEXT_SHORT_TAG_PEP440}@" "${filename}"
    sed_runner "s@rapidsai/devcontainers/features/cuda:[0-9.]*@rapidsai/devcontainers/features/cuda:${NEXT_SHORT_TAG_PEP440}@" "${filename}"
    sed_runner "s@rapidsai/devcontainers/features/rapids-build-utils:[0-9.]*@rapidsai/devcontainers/features/rapids-build-utils:${NEXT_SHORT_TAG_PEP440}@" "${filename}"
    sed_runner "s@rapids-\${localWorkspaceFolderBasename}-[0-9.]*@rapids-\${localWorkspaceFolderBasename}-${NEXT_SHORT_TAG}@g" "${filename}"
done
