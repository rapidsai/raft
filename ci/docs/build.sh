#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.
#################################
# RAFT docs build script for CI #
#################################

if [ -z "$PROJECT_WORKSPACE" ]; then
    echo ">>>> ERROR: Could not detect PROJECT_WORKSPACE in environment"
    echo ">>>> WARNING: This script contains git commands meant for automated building, do not run locally"
    exit 1
fi

export DOCS_WORKSPACE="$WORKSPACE/docs"
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export HOME="$WORKSPACE"
export PROJECT_WORKSPACE=/rapids/raft
export PROJECTS=(raft)

gpuci_logger "Check environment"
env

gpuci_logger "Check GPU usage"
nvidia-smi


gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

gpuci_logger "Check versions"
python --version
$CC --version
$CXX --version

gpuci_logger "Show conda info"
conda info
conda config --show-sources
conda list --show-channel-urls

# Build Doxygen docs
gpuci_logger "Build Doxygen and Sphinx docs"
"$PROJECT_WORKSPACE/build.sh" docs -v

#Commit to Website
cd "$DOCS_WORKSPACE"

for PROJECT in ${PROJECTS[@]}; do
    if [ ! -d "api/$PROJECT/$BRANCH_VERSION" ]; then
        mkdir -p "api/$PROJECT/$BRANCH_VERSION"
    fi
    rm -rf "$DOCS_WORKSPACE/api/$PROJECT/$BRANCH_VERSION/"*
done

mv "$PROJECT_WORKSPACE/docs/_html/"* "$DOCS_WORKSPACE/api/raft/$BRANCH_VERSION"