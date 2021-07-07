#!/bin/bash
# Copyright (c) 2020, NVIDIA CORPORATION.
#####################
# RAFT Style Tester #
#####################

# Ignore errors and set path
set +e
PATH=/opt/conda/bin:$PATH

# Activate common conda env
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids
conda install -c conda-forge clang=8.0.1 clang-tools=8.0.1

# Run flake8 and get results/return code
FLAKE=`flake8 --exclude=cpp,thirdparty,__init__.py,versioneer.py && flake8 --config=python/.flake8.cython`
RETVAL=$?

# Output results if failure otherwise show pass
if [ "$FLAKE" != "" ]; then
  echo -e "\n\n>>>> FAILED: flake8 style check; begin output\n\n"
  echo -e "$FLAKE"
  echo -e "\n\n>>>> FAILED: flake8 style check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: flake8 style check\n\n"
fi

# Check for copyright headers in the files modified currently
COPYRIGHT=`env PYTHONPATH=cpp/scripts python ci/checks/copyright.py 2>&1`
CR_RETVAL=$?
if [ "$RETVAL" = "0" ]; then
  RETVAL=$CR_RETVAL
fi

# Output results if failure otherwise show pass
if [ "$CR_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: copyright check; begin output\n\n"
  echo -e "$COPYRIGHT"
  echo -e "\n\n>>>> FAILED: copyright check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: copyright check\n\n"
fi

# Check for a consistent #include syntax
HASH_INCLUDE=`python cpp/scripts/include_checker.py \
                     cpp/include \
                     cpp/test \
                     2>&1`
HASH_RETVAL=$?
if [ "$RETVAL" = "0" ]; then
  RETVAL=$HASH_RETVAL
fi

# Output results if failure otherwise show pass
if [ "$HASH_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: #include check; begin output\n\n"
  echo -e "$HASH_INCLUDE"
  echo -e "\n\n>>>> FAILED: #include check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: #include check\n\n"
fi

# Check for a consistent code format
FORMAT=`python cpp/scripts/run-clang-format.py 2>&1`
FORMAT_RETVAL=$?
if [ "$RETVAL" = "0" ]; then
  RETVAL=$FORMAT_RETVAL
fi

# Output results if failure otherwise show pass
if [ "$FORMAT_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: clang format check; begin output\n\n"
  echo -e "$FORMAT"
  echo -e "\n\n>>>> FAILED: clang format check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: clang format check\n\n"
fi

exit $RETVAL
