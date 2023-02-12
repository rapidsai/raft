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

echo "Preparing release $CURRENT_TAG => $NEXT_FULL_TAG"

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' $2 && rm -f ${2}.bak
}

sed_runner "s/set(RAPIDS_VERSION .*)/set(RAPIDS_VERSION \"${NEXT_SHORT_TAG}\")/g" cpp/CMakeLists.txt
sed_runner "s/set(RAFT_VERSION .*)/set(RAFT_VERSION \"${NEXT_FULL_TAG}\")/g" cpp/CMakeLists.txt
sed_runner 's/'"pylibraft_version .*)"'/'"pylibraft_version ${NEXT_FULL_TAG})"'/g' python/pylibraft/CMakeLists.txt
sed_runner 's/'"raft_dask_version .*)"'/'"raft_dask_version ${NEXT_FULL_TAG})"'/g' python/raft-dask/CMakeLists.txt
sed_runner 's/'"branch-.*\/RAPIDS.cmake"'/'"branch-${NEXT_SHORT_TAG}\/RAPIDS.cmake"'/g' fetch_rapids.cmake

# Docs update
sed_runner 's/version = .*/version = '"'${NEXT_SHORT_TAG}'"'/g' docs/source/conf.py
sed_runner 's/release = .*/release = '"'${NEXT_FULL_TAG}'"'/g' docs/source/conf.py

for FILE in conda/environments/*.yaml dependencies.yaml; do
  sed_runner "s/dask-cuda=${CURRENT_SHORT_TAG}/dask-cuda=${NEXT_SHORT_TAG}/g" ${FILE};
  sed_runner "s/rapids-build-env=${CURRENT_SHORT_TAG}/rapids-build-env=${NEXT_SHORT_TAG}/g" ${FILE};
  sed_runner "s/rapids-doc-env=${CURRENT_SHORT_TAG}/rapids-doc-env=${NEXT_SHORT_TAG}/g" ${FILE};
  sed_runner "s/rapids-notebook-env=${CURRENT_SHORT_TAG}/rapids-notebook-env=${NEXT_SHORT_TAG}/g" ${FILE};
  sed_runner "s/rmm=${CURRENT_SHORT_TAG}/rmm=${NEXT_SHORT_TAG}/g" ${FILE};
  sed_runner "s/ucx-py=.*/ucx-py=${NEXT_UCX_PY_VERSION}/g" ${FILE};
done

sed_runner "/^ucx_py_version:$/ {n;s/.*/  - \"${NEXT_UCX_PY_VERSION}\"/}" conda/recipes/raft-dask/conda_build_config.yaml

# Wheel builds install dask-cuda from source, update its branch
for FILE in .github/workflows/*.yaml; do
  sed_runner "s/dask-cuda.git@branch-[^\"\s]\+/dask-cuda.git@branch-${NEXT_SHORT_TAG}/g" ${FILE};
done

# Need to distutils-normalize the original version
NEXT_SHORT_TAG_PEP440=$(python -c "from setuptools.extern import packaging; print(packaging.version.Version('${NEXT_SHORT_TAG}'))")
NEXT_UCX_PY_SHORT_TAG_PEP440=$(python -c "from setuptools.extern import packaging; print(packaging.version.Version('${NEXT_UCX_PY_SHORT_TAG}'))")

# Wheel builds install intra-RAPIDS dependencies from same release
sed_runner "s/{cuda_suffix}[^\"].*\",/{cuda_suffix}==${NEXT_SHORT_TAG_PEP440}.*\",/g" python/pylibraft/setup.py
sed_runner "s/{cuda_suffix}.*\"\]/{cuda_suffix}==${NEXT_SHORT_TAG_PEP440}.*\"\]/g" python/pylibraft/_custom_build/backend.py
sed_runner "s/dask-cuda==.*\",/dask-cuda==${NEXT_SHORT_TAG_PEP440}.*\",/g" python/raft-dask/setup.py
sed_runner "s/pylibraft{cuda_suffix}.*\",/pylibraft{cuda_suffix}==${NEXT_SHORT_TAG_PEP440}.*\",/g" python/raft-dask/setup.py
sed_runner "s/ucx-py{cuda_suffix}.*\",/ucx-py{cuda_suffix}==${NEXT_UCX_PY_SHORT_TAG_PEP440}.*\",/g" python/raft-dask/setup.py

for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-action-workflows/ s/@.*/@branch-${NEXT_SHORT_TAG}/g" "${FILE}"
done
