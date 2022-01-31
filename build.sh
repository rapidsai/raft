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

VALIDARGS="clean libraft pyraft docs -v -g --compile-libs --compile-nn --compile-dist --allgpuarch --nvtx --show_depr_warn -h --nogtest --buildfaiss"
HELP="$0 [<target> ...] [<flag> ...]
 where <target> is:
   clean            - remove all existing build artifacts and configuration (start over)
   libraft          - build the raft C++ code only. Also builds the C-wrapper library
                      around the C++ code.
   pyraft             - build the cuml Python package
   docs             - build the documentation

 and <flag> is:
   -v               - verbose build mode
   -g               - build for debug
   --compile-libs    - compile shared libraries for all components
   --compile-nn     - compile shared library for nn component
   --compile-dist   - compile shared library for distance component
   --allgpuarch     - build for all supported GPU architectures
   --buildfaiss     - build faiss statically into raft
   --nogtest        - do not build google tests for libraft
   --nvtx           - Enable nvtx for profiling support
   --show_depr_warn - show cmake deprecation warnings
   -h               - print this text

 default action (no args) is to build both libraft and pyraft targets
"
CPP_RAFT_BUILD_DIR=${REPODIR}/cpp/build
SPHINX_BUILD_DIR=${REPODIR}/docs
PY_RAFT_BUILD_DIR=${REPODIR}/python/build
PYTHON_DEPS_CLONE=${REPODIR}/python/external_repositories
BUILD_DIRS="${CPP_RAFT_BUILD_DIR} ${PY_RAFT_BUILD_DIR} ${PYTHON_DEPS_CLONE}"

# Set defaults for vars modified by flags to this script
CMAKE_LOG_LEVEL=""
VERBOSE_FLAG=""
BUILD_ALL_GPU_ARCH=0
BUILD_TESTS=YES
BUILD_STATIC_FAISS=OFF
COMPILE_LIBRARIES=${BUILD_TESTS}
ENABLE_NN_DEPENDENCIES=${BUILD_TESTS}
SINGLEGPU=""
NVTX=OFF
CLEAN=0
DISABLE_DEPRECATION_WARNINGS=ON
CMAKE_TARGET=""

# Set defaults for vars that may not have been defined externally
#  FIXME: if INSTALL_PREFIX is not set, check PREFIX, then check
#         CONDA_PREFIX, but there is no fallback from there!
INSTALL_PREFIX=${INSTALL_PREFIX:=${PREFIX:=${CONDA_PREFIX}}}
PARALLEL_LEVEL=${PARALLEL_LEVEL:=""}
BUILD_ABI=${BUILD_ABI:=ON}

# Default to Ninja if generator is not specified
export CMAKE_GENERATOR="${CMAKE_GENERATOR:=Ninja}"

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
    VERBOSE_FLAG=-v
    CMAKE_LOG_LEVEL="--log-level=VERBOSE"
    set -x
fi
if hasArg -g; then
    BUILD_TYPE=Debug
fi

if hasArg --allgpuarch; then
    BUILD_ALL_GPU_ARCH=1
fi
if hasArg --nogtest; then
    BUILD_TESTS=OFF
    COMPILE_LIBRARIES=OFF
    ENABLE_NN_DEPENDENCIES=OFF
fi

if hasArg --compile-nn || hasArg --compile-libs; then
    ENABLE_NN_DEPENDENCIES=ON
    COMPILE_LIBRARIES=ON
    CMAKE_TARGET="raft_nn_lib;${CMAKE_TARGET}"
fi
if hasArg --compile-dist || hasArg --compile-libs; then
    COMPILE_LIBRARIES=ON
    CMAKE_TARGET="raft_distance_lib;${CMAKE_TARGET}"
fi

if hasArg --buildfaiss; then
    BUILD_STATIC_FAISS=ON
fi
if hasArg --singlegpu; then
    SINGLEGPU="--singlegpu"
fi
if hasArg --nvtx; then
    NVTX=ON
fi
if hasArg --show_depr_warn; then
    DISABLE_DEPRECATION_WARNINGS=OFF
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
if (( ${NUMARGS} == 0 )) || hasArg libraft || hasArg docs; then
    if (( ${BUILD_ALL_GPU_ARCH} == 0 )); then
        RAFT_CMAKE_CUDA_ARCHITECTURES="NATIVE"
        echo "Building for the architecture of the GPU in the system..."
    else
        RAFT_CMAKE_CUDA_ARCHITECTURES="ALL"
        echo "Building for *ALL* supported GPU architectures..."
    fi

    cmake -S ${REPODIR}/cpp -B ${CPP_RAFT_BUILD_DIR} ${CMAKE_LOG_LEVEL} \
          -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
          -DCMAKE_CUDA_ARCHITECTURES=${RAFT_CMAKE_CUDA_ARCHITECTURES} \
          -DRAFT_COMPILE_LIBRARIES=${COMPILE_LIBRARIES} \
          -DRAFT_ENABLE_NN_DEPENDENCIES=${ENABLE_NN_DEPENDENCIES} \
          -DNVTX=${NVTX} \
          -DDISABLE_DEPRECATION_WARNINGS=${DISABLE_DEPRECATION_WARNINGS} \
          -DBUILD_TESTS=${BUILD_TESTS} \
          -DRAFT_USE_FAISS_STATIC=${BUILD_STATIC_FAISS} \
          ..

  if (( ${NUMARGS} == 0 )) || hasArg libraft; then
      # Run all c++ targets at once
      if hasArg --compile-nn || hasArg --compile-dist || hasArg --compile-libs; then
        if ! hasArg --nogtest; then
          CMAKE_TARGET="test_raft;${CMAKE_TARGET}"
        fi

        echo "-- Compiling targets: ${CMAKE_TARGET}"
        cmake --build  ${CPP_RAFT_BUILD_DIR} -j${PARALLEL_LEVEL} ${VERBOSE_FLAG} --target ${CMAKE_TARGET}
      else
        cmake --build  ${CPP_RAFT_BUILD_DIR} -j${PARALLEL_LEVEL} ${VERBOSE_FLAG}
      fi
  fi

fi

# Build and (optionally) install the cuml Python package
if (( ${NUMARGS} == 0 )) || hasArg pyraft || hasArg docs; then

    cd ${REPODIR}/python
    if [[ ${INSTALL_TARGET} != "" ]]; then
        python setup.py build_ext -j${PARALLEL_LEVEL:-1} --inplace ${SINGLEGPU}
    else
        python setup.py build_ext -j${PARALLEL_LEVEL:-1} --inplace --library-dir=${LIBRAFT_BUILD_DIR} ${SINGLEGPU}
    fi
fi

if hasArg docs; then
    cmake --build ${CPP_RAFT_BUILD_DIR} --target docs_raft
    cd ${SPHINX_BUILD_DIR}
    make html
fi
