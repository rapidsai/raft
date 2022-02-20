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

VALIDARGS="clean libraft pyraft docs -v -g --noinstall --compile-libs --compile-nn --compile-dist --allgpuarch --nvtx --show_depr_warn -h --nogtest --buildfaiss"
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
   --noinstall     - do not install cmake targets
   --nvtx           - Enable nvtx for profiling support
   --show_depr_warn - show cmake deprecation warnings
   -h               - print this text

 default action (no args) is to build both libraft and pyraft targets
"
LIBRAFT_BUILD_DIR=${LIBRAFT_BUILD_DIR:=${REPODIR}/cpp/build}
SPHINX_BUILD_DIR=${REPODIR}/docs
PY_RAFT_BUILD_DIR=${REPODIR}/python/build
PYTHON_DEPS_CLONE=${REPODIR}/python/external_repositories
BUILD_DIRS="${LIBRAFT_BUILD_DIR} ${PY_RAFT_BUILD_DIR} ${PYTHON_DEPS_CLONE}"

# Set defaults for vars modified by flags to this script
CMAKE_LOG_LEVEL=""
VERBOSE_FLAG=""
BUILD_ALL_GPU_ARCH=0
BUILD_TESTS=YES
BUILD_STATIC_FAISS=OFF
COMPILE_LIBRARIES=OFF
COMPILE_NN_LIBRARY=OFF
COMPILE_DIST_LIBRARY=OFF
ENABLE_NN_DEPENDENCIES=${BUILD_TESTS}
NVTX=OFF
CLEAN=0
DISABLE_DEPRECATION_WARNINGS=ON
CMAKE_TARGET=""
INSTALL_TARGET="install"

# Set defaults for vars that may not have been defined externally
#  FIXME: if INSTALL_PREFIX is not set, check PREFIX, then check
#         CONDA_PREFIX, but there is no fallback from there!
INSTALL_PREFIX=${INSTALL_PREFIX:=${PREFIX:=${CONDA_PREFIX}}}
PARALLEL_LEVEL=${PARALLEL_LEVEL:=`nproc`}
BUILD_ABI=${BUILD_ABI:=ON}

# Default to Ninja if generator is not specified
export CMAKE_GENERATOR="${CMAKE_GENERATOR:=Ninja}"

function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

if hasArg --noinstall; then
    INSTALL_TARGET=""
fi

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
    VERBOSE_FLAG="-v"
    CMAKE_LOG_LEVEL="VERBOSE"
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

if hasArg --compile-libs; then
  COMPILE_LIBRARIES=ON
fi

if hasArg --compile-nn || hasArg --compile-libs; then
    ENABLE_NN_DEPENDENCIES=ON
    COMPILE_NN_LIBRARY=ON
    CMAKE_TARGET="raft_nn_lib;${CMAKE_TARGET}"
fi
if hasArg --compile-dist || hasArg --compile-libs; then
    COMPILE_DIST_LIBRARY=ON
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

    mkdir -p ${LIBRAFT_BUILD_DIR}
    cd ${LIBRAFT_BUILD_DIR}
    cmake -S ${REPODIR}/cpp -B ${LIBRAFT_BUILD_DIR} \
          -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
          -DCMAKE_CUDA_ARCHITECTURES=${RAFT_CMAKE_CUDA_ARCHITECTURES} \
          -DRAFT_COMPILE_LIBRARIES=${COMPILE_LIBRARIES} \
          -DRAFT_ENABLE_NN_DEPENDENCIES=${ENABLE_NN_DEPENDENCIES} \
          -DNVTX=${NVTX} \
          -DDISABLE_DEPRECATION_WARNINGS=${DISABLE_DEPRECATION_WARNINGS} \
          -DBUILD_TESTS=${BUILD_TESTS} \
          -DCMAKE_MESSAGE_LOG_LEVEL=${CMAKE_LOG_LEVEL} \
          -DRAFT_COMPILE_NN_LIBRARY=${COMPILE_NN_LIBRARY} \
          -DRAFT_COMPILE_DIST_LIBRARY=${COMPILE_DIST_LIBRARY} \
          -DRAFT_USE_FAISS_STATIC=${BUILD_STATIC_FAISS}

  if (( ${NUMARGS} == 0 )) || hasArg libraft; then
      # Run all c++ targets at once
      if ! hasArg --nogtest; then
        CMAKE_TARGET="${CMAKE_TARGET};test_raft;"
      fi

      echo "-- Compiling targets: ${CMAKE_TARGET}, verbose=${VERBOSE_FLAG}"
      cmake --build  "${LIBRAFT_BUILD_DIR}" ${VERBOSE_FLAG} -j${PARALLEL_LEVEL} --target ${CMAKE_TARGET} ${INSTALL_TARGET}
  fi
fi

# Build and (optionally) install the cuml Python package
if (( ${NUMARGS} == 0 )) || hasArg pyraft || hasArg docs; then

    cd ${REPODIR}/python
    if [[ ${INSTALL_TARGET} != "" ]]; then
        python setup.py build_ext -j${PARALLEL_LEVEL:-1} --inplace --library-dir=${LIBRAFT_BUILD_DIR} install --single-version-externally-managed --record=record.txt
    else
        python setup.py build_ext -j${PARALLEL_LEVEL:-1} --inplace --library-dir=${LIBRAFT_BUILD_DIR}
    fi
fi

if hasArg docs; then
    cmake --build ${LIBRAFT_BUILD_DIR} --target docs_raft
    cd ${SPHINX_BUILD_DIR}
    make html
fi
