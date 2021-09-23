# RAFT Build and Development Guide

- [Usage of RAFT by downstream projects](#usage-of-raft-by-downstream-projects)
    - [C++ Integration](#c-integration)
    - [Python/Cython Integration](#pythoncython-integration)
- [Building and running tests](#building-and-running-tests)
- [CI Process](#ci-process)
- [Developer Guide](#developer-guide)
    - [Local Development](#local-development)
    - [Submitting PRs](#submitting-prs)



## Usage of RAFT by downstream projects

### C++ Integration

C++ RAFT is a header only library, so it can be easily configured using CMake by consuming libraries. Since this repo is intended to be included by downstream repos, the recommended way of accomplishing that is using CMake's git cloning functionality:


```cmake
if(DEFINED ENV{RAFT_PATH})
  message(STATUS "RAFT_PATH environment variable detected.")
  message(STATUS "RAFT_DIR set to $ENV{RAFT_PATH}")
  set(RAFT_DIR ENV{RAFT_PATH})

else(DEFINED ENV{RAFT_PATH})
  message(STATUS "RAFT_PATH environment variable NOT detected, cloning RAFT")
  set(RAFT_GIT_DIR ${CMAKE_CURRENT_BINARY_DIR}/raft CACHE STRING "Path to RAFT repo")

  ExternalProject_Add(raft
    GIT_REPOSITORY    git@github.com:rapidsai/raft.git
    GIT_TAG           pinned_commit/git_tag/branch
    PREFIX            ${RAFT_GIT_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   "")

  set(RAFT_INCLUDE_DIR ${RAFT_GIT_DIR}/src/raft/cpp/include CACHE STRING "RAFT include variable")
endif(DEFINED ENV{RAFT_PATH})

```

This create the variable `$RAFT_INCLUDE_DIR` variable that can be used in `include_directories`, and then the related header files can be included when needed.

### Python/Cython Integration

RAFT's Python and Cython code have been designed to be included in projects that use RAFT, as opposed to be distributed by itself as a Python package. To use:

- The file `setuputils.py` is included in RAFT's `python` folder. Copy the file to your repo, in a location where it can be imported by `setup.py`
- In your setup.py, use the function `use_raft_package`, for example for cuML:


```python
# Optional location of C++ build folder that can be configured by the user
libcuml_path = get_environment_option('CUML_BUILD_PATH')
# Optional location of RAFT that can be confugred by the user
raft_path = get_environment_option('RAFT_PATH')

use_raft_package(raft_path, libcuml_path)
```

The usage of RAFT by the consuming repo's python code follows the rules:
1. If the environment variable `RAFT_PATH` points to the RAFT repo, then that will be used.
2. If there is a C++ build folder that has cloned RAFT already, setup.py will use that RAFT.
3. If none of the above happened, then setup.py will clone RAFT and use it directly.

- After `setup.py` calls the `use_raft_package` function, the RAFT python code will be included (via a symlink) in the consuming repo package, under a raft subfolder. So for example, `cuml` python package includes RAFT in `cuml.raft`.


## Building and running tests

Since RAFT is not meant to create any artifact on itself, but be included in other projects, the build infrastructure is focused only on testing.

The base folder in the repository contains a `build.sh` script that builds both the C++ and Python code, which is the recommended way of building the tests.

To run C++ tests:

```bash
./test_raft
```

To run Python tests, if `install` setup.py target is not run:

```bash
cd python
python -m pytest raft
```

To build manually, you can also use `CMake` and setup.py directly. For C++:

```bash
cd cpp
mkdir build
cd build
cmake ..
```

There is no `install` target currently.

For python:

```bash
cd python
python setup.py build_ext --inplace
```


## CI Process

PRs submitted to RAFT will always run the RAFT tests (once GPUCI is enabled). Additionally, RAFT has convenience functionality to run tests of the following projects that use RAFT: cuML and cuGraph.

To run these other tests, turn `ON` the variables in `ci/prtest.config` in your PR:

```bash
RUN_CUGRAPH_LIBCUGRAPH_TESTS=OFF
RUN_CUGRAPH_PYTHON_TESTS=OFF

RUN_CUML_LIBCUML_TESTS=OFF
RUN_CUML_PRIMS_TESTS=OFF
RUN_CUML_PYTHON_TESTS=OFF
```

This will make it so that CI in the PR will clone and build the respective repository, but the repository **will be built using the fork/branch of RAFT in the PR**. This allows to test changes in RAFT without the need of opening PRs in the other repositories.

Before merging the PR, those variables need to be returned to `OFF`.


## Developer Guide

### Local Development

To help working with RAFT and consuming projects as seamless as possible, this section describes how a typical workflow looks like and gives some guidelines for developers working in projects that affect code in both RAFT and at least one downstream repository.

Using as an example developer working on cuML and RAFT, we recommend the following:

- Create two working folders: one containing the cloned cuML repository and the other the cloned RAFT one.
- Create environment variable `RAFT_PATH` pointing to the location of the RAFT path.
- Work on same named branches in both repos/folders.

This will facilitate development, and the `RAFT_PATH` variable will make it so that the downstream repository, in this case cuML, builds using the locally cloned RAFT (as descrbed in the first step).

### Submitting PRs Guidelines

If you have changes to both RAFT and at least one downstream repo, then:

- It is recommended to open a PR to both repositories (for visibility and CI tests).
- Change the pinned branch/commit in the downstream repo PR to point to the fork and branch used for the RAFT PR to make CI run tests
- If your changes might affect usage of RAFT by other downnstream repos, alert reviewers and open a github issue or PR in that downstream repo as approproate.
- The PR to RAFT will be merged first, so that the downstream repo PR pinned branch/commit can be returned to the main RAFT branch and run CI with it.
