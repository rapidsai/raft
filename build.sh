#!/bin/bash

# Copyright (c) 2020-2022, NVIDIA CORPORATION.

# raft build script

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

VALIDARGS="clean libraft pylibraft raft-dask docs tests bench clean -v -g --install --compile-libs --compile-nn --compile-dist --allgpuarch --no-nvtx --show_depr_warn -h --buildfaiss --minimal-deps"
HELP="$0 [<target> ...] [<flag> ...] [--cmake-args=\"<args>\"] [--cache-tool=<tool>]
 where <target> is:
   clean            - remove all existing build artifacts and configuration (start over)
   libraft          - build the raft C++ code only. Also builds the C-wrapper library
                      around the C++ code.
   pylibraft        - build the pylibraft Python package
   raft-dask        - build the raft-dask Python package. this also requires pylibraft.
   docs             - build the documentation
   tests            - build the tests
   bench            - build the benchmarks

 and <flag> is:
   -v                          - verbose build mode
   -g                          - build for debug
   --compile-libs              - compile shared libraries for all components
   --compile-nn                - compile shared library for nn component
   --compile-dist              - compile shared library for distance and current random components
                                 (eventually, this will be renamed to something more generic and
                                  the only option to be supported)
   --minimal-deps              - disables dependencies like thrust so they can be overridden.
                                 can be useful for a pure header-only install
   --allgpuarch                - build for all supported GPU architectures
   --buildfaiss                - build faiss statically into raft
   --install                   - install cmake targets
   --no-nvtx                   - disable nvtx (profiling markers), but allow enabling it in downstream projects
   --show_depr_warn            - show cmake deprecation warnings
   --cmake-args=\\\"<args>\\\" - pass arbitrary list of CMake configuration options (escape all quotes in argument)
   --cache-tool=<tool>         - pass the build cache tool (eg: ccache, sccache, distcc) that will be used
                                 to speedup the build process.
   -h                          - print this text

 default action (no args) is to build both libraft and raft-dask targets
"
LIBRAFT_BUILD_DIR=${LIBRAFT_BUILD_DIR:=${REPODIR}/cpp/build}
SPHINX_BUILD_DIR=${REPODIR}/docs
PY_RAFT_BUILD_DIR=${REPODIR}/python/raft/build
PY_LIBRAFT_BUILD_DIR=${REPODIR}/python/pylibraft/_skbuild
BUILD_DIRS="${LIBRAFT_BUILD_DIR} ${PY_RAFT_BUILD_DIR} ${PY_LIBRAFT_BUILD_DIR}"

# Set defaults for vars modified by flags to this script
CMAKE_LOG_LEVEL=""
VERBOSE_FLAG=""
BUILD_ALL_GPU_ARCH=0
BUILD_TESTS=OFF
BUILD_BENCH=OFF
BUILD_STATIC_FAISS=OFF
COMPILE_LIBRARIES=OFF
COMPILE_NN_LIBRARY=OFF
COMPILE_DIST_LIBRARY=OFF
ENABLE_NN_DEPENDENCIES=OFF

ENABLE_thrust_DEPENDENCY=ON

CACHE_ARGS=""
NVTX=ON
CLEAN=0
UNINSTALL=0
DISABLE_DEPRECATION_WARNINGS=ON
CMAKE_TARGET=""
INSTALL_TARGET=""

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

function cmakeArgs {
    # Check for multiple cmake args options
    if [[ $(echo $ARGS | { grep -Eo "\-\-cmake\-args" || true; } | wc -l ) -gt 1 ]]; then
        echo "Multiple --cmake-args options were provided, please provide only one: ${ARGS}"
        exit 1
    fi

    # Check for cmake args option
    if [[ -n $(echo $ARGS | { grep -E "\-\-cmake\-args" || true; } ) ]]; then
        # There are possible weird edge cases that may cause this regex filter to output nothing and fail silently
        # the true pipe will catch any weird edge cases that may happen and will cause the program to fall back
        # on the invalid option error
        EXTRA_CMAKE_ARGS=$(echo $ARGS | { grep -Eo "\-\-cmake\-args=\".+\"" || true; })
        if [[ -n ${EXTRA_CMAKE_ARGS} ]]; then
            # Remove the full  EXTRA_CMAKE_ARGS argument from list of args so that it passes validArgs function
            ARGS=${ARGS//$EXTRA_CMAKE_ARGS/}
            # Filter the full argument down to just the extra string that will be added to cmake call
            EXTRA_CMAKE_ARGS=$(echo $EXTRA_CMAKE_ARGS | grep -Eo "\".+\"" | sed -e 's/^"//' -e 's/"$//')
        fi
    fi
}

function cacheTool {
    # Check for multiple cache options
    if [[ $(echo $ARGS | { grep -Eo "\-\-cache\-tool" || true; } | wc -l ) -gt 1 ]]; then
        echo "Multiple --cache-tool options were provided, please provide only one: ${ARGS}"
        exit 1
    fi
    # Check for cache tool option
    if [[ -n $(echo $ARGS | { grep -E "\-\-cache\-tool" || true; } ) ]]; then
        # There are possible weird edge cases that may cause this regex filter to output nothing and fail silently
        # the true pipe will catch any weird edge cases that may happen and will cause the program to fall back
        # on the invalid option error
        CACHE_TOOL=$(echo $ARGS | sed -e 's/.*--cache-tool=//' -e 's/ .*//')
        if [[ -n ${CACHE_TOOL} ]]; then
            # Remove the full CACHE_TOOL argument from list of args so that it passes validArgs function
            ARGS=${ARGS//--cache-tool=$CACHE_TOOL/}
            CACHE_ARGS="-DCMAKE_CUDA_COMPILER_LAUNCHER=${CACHE_TOOL} -DCMAKE_C_COMPILER_LAUNCHER=${CACHE_TOOL} -DCMAKE_CXX_COMPILER_LAUNCHER=${CACHE_TOOL}"
        fi
    fi
}

if hasArg -h || hasArg --help; then
    echo "${HELP}"
    exit 0
fi

# Check for valid usage
if (( ${NUMARGS} != 0 )); then
    cmakeArgs
    cacheTool
    for a in ${ARGS}; do
        if ! (echo " ${VALIDARGS} " | grep -q " ${a} "); then
            echo "Invalid option: ${a}"
            exit 1
        fi
    done
fi

# Process flags
if hasArg --install; then
    INSTALL_TARGET="install"
fi

if hasArg --minimal-deps; then
    ENABLE_thrust_DEPENDENCY=OFF
fi

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

if hasArg --compile-libs || (( ${NUMARGS} == 0 )); then
    COMPILE_LIBRARIES=ON
fi

if hasArg --compile-nn || hasArg --compile-libs || (( ${NUMARGS} == 0 )); then
    ENABLE_NN_DEPENDENCIES=ON
    COMPILE_NN_LIBRARY=ON
    CMAKE_TARGET="${CMAKE_TARGET};raft_nn_lib"
fi

if hasArg --compile-dist || hasArg --compile-libs || hasArg pylibraft || (( ${NUMARGS} == 0 )); then
    COMPILE_DIST_LIBRARY=ON
    CMAKE_TARGET="${CMAKE_TARGET};raft_distance_lib"
fi

if hasArg tests || (( ${NUMARGS} == 0 )); then
    BUILD_TESTS=ON
    COMPILE_DIST_LIBRARY=ON
    ENABLE_NN_DEPENDENCIES=ON
    COMPILE_NN_LIBRARY=ON
    CMAKE_TARGET="${CMAKE_TARGET};test_raft"
fi

if hasArg bench || (( ${NUMARGS} == 0 )); then
    BUILD_BENCH=ON
    ENABLE_NN_DEPENDENCIES=ON
    CMAKE_TARGET="${CMAKE_TARGET};bench_raft"
fi

if hasArg --buildfaiss; then
    BUILD_STATIC_FAISS=ON
fi
if hasArg --no-nvtx; then
    NVTX=OFF
fi
if hasArg --show_depr_warn; then
    DISABLE_DEPRECATION_WARNINGS=OFF
fi
if hasArg clean; then
    CLEAN=1
fi
if hasArg uninstall; then
    UNINSTALL=1
fi

if [[ ${CMAKE_TARGET} == "" ]]; then
    CMAKE_TARGET="all"
fi

# Append `-DFIND_RAFT_CPP=ON` to EXTRA_CMAKE_ARGS unless a user specified the option.
if [[ "${EXTRA_CMAKE_ARGS}" != *"DFIND_RAFT_CPP"* ]]; then
    EXTRA_CMAKE_ARGS="${EXTRA_CMAKE_ARGS} -DFIND_RAFT_CPP=ON"
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

    cd ${REPODIR}/python/raft-dask
    python setup.py clean --all
    cd ${REPODIR}

    cd ${REPODIR}/python/pylibraft
    python setup.py clean --all
    cd ${REPODIR}
fi

################################################################################
# Configure for building all C++ targets
if (( ${NUMARGS} == 0 )) || hasArg libraft || hasArg pylibraft || hasArg docs || hasArg tests || hasArg bench; then
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
          -DRAFT_NVTX=${NVTX} \
          -DDISABLE_DEPRECATION_WARNINGS=${DISABLE_DEPRECATION_WARNINGS} \
          -DBUILD_TESTS=${BUILD_TESTS} \
          -DBUILD_BENCH=${BUILD_BENCH} \
          -DCMAKE_MESSAGE_LOG_LEVEL=${CMAKE_LOG_LEVEL} \
          -DRAFT_COMPILE_NN_LIBRARY=${COMPILE_NN_LIBRARY} \
          -DRAFT_COMPILE_DIST_LIBRARY=${COMPILE_DIST_LIBRARY} \
          -DRAFT_USE_FAISS_STATIC=${BUILD_STATIC_FAISS} \
          -DRAFT_ENABLE_thrust_DEPENDENCY=${ENABLE_thrust_DEPENDENCY} \
          ${CACHE_ARGS} \
          ${EXTRA_CMAKE_ARGS}

  if [[ ${CMAKE_TARGET} != "" ]]; then
      echo "-- Compiling targets: ${CMAKE_TARGET}, verbose=${VERBOSE_FLAG}"
      if [[ ${INSTALL_TARGET} != "" ]]; then
        cmake --build  "${LIBRAFT_BUILD_DIR}" ${VERBOSE_FLAG} -j${PARALLEL_LEVEL} --target ${CMAKE_TARGET} ${INSTALL_TARGET}
      else
        cmake --build  "${LIBRAFT_BUILD_DIR}" ${VERBOSE_FLAG} -j${PARALLEL_LEVEL} --target ${CMAKE_TARGET}
      fi
  fi
fi

# Build and (optionally) install the raft-dask Python package
if (( ${NUMARGS} == 0 )) || hasArg raft-dask || hasArg docs; then

    cd ${REPODIR}/python/raft-dask
    python setup.py build_ext --inplace -- -DCMAKE_PREFIX_PATH="${LIBRAFT_BUILD_DIR};${INSTALL_PREFIX}" -DCMAKE_LIBRARY_PATH=${LIBRAFT_BUILD_DIR} ${EXTRA_CMAKE_ARGS} -- -j${PARALLEL_LEVEL:-1}
    if [[ ${INSTALL_TARGET} != "" ]]; then
        python setup.py install --single-version-externally-managed --record=record.txt -- -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX} ${EXTRA_CMAKE_ARGS}
    fi
fi

# Build and (optionally) install the pylibraft Python package
if (( ${NUMARGS} == 0 )) || hasArg pylibraft; then

    cd ${REPODIR}/python/pylibraft
    python setup.py build_ext --inplace -- -DCMAKE_PREFIX_PATH="${LIBRAFT_BUILD_DIR};${INSTALL_PREFIX}" -DCMAKE_LIBRARY_PATH=${LIBRAFT_BUILD_DIR} ${EXTRA_CMAKE_ARGS} -- -j${PARALLEL_LEVEL:-1}
    if [[ ${INSTALL_TARGET} != "" ]]; then
        python setup.py install --single-version-externally-managed --record=record.txt -- -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX} ${EXTRA_CMAKE_ARGS}
    fi
fi

if hasArg docs; then
    cmake --build ${LIBRAFT_BUILD_DIR} --target docs_raft
    cd ${SPHINX_BUILD_DIR}
    make html
fi
