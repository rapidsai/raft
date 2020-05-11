#!/bin/bash

# Copyright (c) 2020, NVIDIA CORPORATION.

# cuml build script

# This script is used to build the component(s) in this repo from
# source, and can be called with various options to customize the
# build as needed (see the help output for details)

# Abort script on first error
set -e

NUMARGS=$#
ARGS=$*

# NOTE: ensure all dir changes are relative to the location of this
# script, and that this script resides in the repo dir!
REPODIR=$(cd $(dirname $0); pwd)

VALIDARGS="clean cppraft pyraft -v -g --allgpuarch --nvtx --show_depr_warn -h"
HELP="$0 [<target> ...] [<flag> ...]
 where <target> is:
   clean            - remove all existing build artifacts and configuration (start over)
   cppraft          - build the cuml C++ code only. Also builds the C-wrapper library
                      around the C++ code.
   pyraft             - build the cuml Python package
 and <flag> is:
   -v               - verbose build mode
   -g               - build for debug
   --allgpuarch     - build for all supported GPU architectures
   --nvtx           - Enable nvtx for profiling support
   --show_depr_warn - show cmake deprecation warnings
   -h               - print this text

 default action (no args) is to build both cppraft and pyraft targets
"
CPP_RAFT_BUILD_DIR=${REPODIR}/cpp/build
PY_RAFT_BUILD_DIR=${REPODIR}/python/build
PYTHON_DEPS_CLONE=${REPODIR}/python/external_repositories
BUILD_DIRS="${CPP_RAFT_BUILD_DIR} ${PY_RAFT_BUILD_DIR} ${PYTHON_DEPS_CLONE}"

# Set defaults for vars modified by flags to this script
VERBOSE=""
BUILD_ALL_GPU_ARCH=0
SINGLEGPU=""
NVTX=OFF
CLEAN=0
BUILD_DISABLE_DEPRECATION_WARNING=ON

# Set defaults for vars that may not have been defined externally
#  FIXME: if INSTALL_PREFIX is not set, check PREFIX, then check
#         CONDA_PREFIX, but there is no fallback from there!
INSTALL_PREFIX=${INSTALL_PREFIX:=${PREFIX:=${CONDA_PREFIX}}}
PARALLEL_LEVEL=${PARALLEL_LEVEL:=""}
BUILD_ABI=${BUILD_ABI:=ON}

function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

if hasArg -h || hasArg --help; then
    echo "${HELP}"
    exit 0
fi

# Check for valid usage
if (( ${NUMARGS} != 0 )); then
    for a in ${ARGS}; do
  if ! (echo " ${VALIDARGS} " | grep -q " ${a} "); then
      echo "Invalid option: ${a}"
      exit 1
  fi
    done
fi

# Process flags
if hasArg -v; then
    VERBOSE=1
fi
if hasArg -g; then
    BUILD_TYPE=Debug
fi

if hasArg --allgpuarch; then
    BUILD_ALL_GPU_ARCH=1
fi
if hasArg --singlegpu; then
    SINGLEGPU="--singlegpu"
fi
if hasArg --nvtx; then
    NVTX=ON
fi
if hasArg --show_depr_warn; then
    BUILD_DISABLE_DEPRECATION_WARNING=OFF
fi
if hasArg clean; then
    CLEAN=1
fi

# If clean given, run it prior to any other steps
if (( ${CLEAN} == 1 )); then
    # If the dirs to clean are mounted dirs in a container, the
    # contents should be removed but the mounted dirs will remain.
    # The find removes all contents but leaves the dirs, the rmdir
    # attempts to remove the dirs but can fail safely.
    for bd in ${BUILD_DIRS}; do
      if [ -d ${bd} ]; then
          find ${bd} -mindepth 1 -delete
          rmdir ${bd} || true
      fi

    done

    cd ${REPODIR}/python
    python setup.py clean --all
    cd ${REPODIR}
fi

################################################################################
# Configure for building all C++ targets
if (( ${NUMARGS} == 0 )) || hasArg cppraft; then
    if (( ${BUILD_ALL_GPU_ARCH} == 0 )); then
        GPU_ARCH=""
        echo "Building for the architecture of the GPU in the system..."
    else
        GPU_ARCH="-DGPU_ARCHS=ALL"
        echo "Building for *ALL* supported GPU architectures..."
    fi

    mkdir -p ${CPP_RAFT_BUILD_DIR}
    cd ${CPP_RAFT_BUILD_DIR}

    cmake -DNVTX=${NVTX} \
          -DPARALLEL_LEVEL=${PARALLEL_LEVEL} \
          -DNCCL_PATH=${INSTALL_PREFIX} \
          -DDISABLE_DEPRECATION_WARNING=${BUILD_DISABLE_DEPRECATION_WARNING} \
          ..

fi

# Run all make targets at once

MAKE_TARGETS=
if hasArg cppraft; then
    MAKE_TARGETS="${MAKE_TARGETS} test_raft"
fi


# If `./build.sh pyraft` is called, don't build C/C++ components
if (( ${NUMARGS} == 0 )) || hasArg cppraft; then
# If there are no targets specified when calling build.sh, it will
# just call `make -j`. This avoids a lot of extra printing
    cd ${CPP_RAFT_BUILD_DIR}
    make -j${PARALLEL_LEVEL} ${MAKE_TARGETS} VERBOSE=${VERBOSE}

fi


# Build and (optionally) install the cuml Python package
if (( ${NUMARGS} == 0 )) || hasArg pyraft; then

    cd ${REPODIR}/python
    if [[ ${INSTALL_TARGET} != "" ]]; then
        python setup.py build_ext -j${PARALLEL_LEVEL:-1} --inplace ${SINGLEGPU}
    else
        python setup.py build_ext -j${PARALLEL_LEVEL:-1} --inplace --library-dir=${LIBCUML_BUILD_DIR} ${SINGLEGPU}
    fi
fi
