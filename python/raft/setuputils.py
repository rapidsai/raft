#
# Copyright (c) 2018-2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import glob
import os
import shutil
import sys


def get_environment_option(name):
    ENV_VARIABLE = os.environ.get(name, False)

    if not ENV_VARIABLE:
        print("-- " + name + " environment variable not set.")

    else:
        print("-- " + name + " detected with value: " + str(ENV_VARIABLE))

    return ENV_VARIABLE


def get_cli_option(name):
    if name in sys.argv:
        print("-- Detected " + str(name) + " build option.")
        return True

    else:
        return False


def clean_folder(path):
    """
    Function to clean all Cython and Python artifacts and cache folders. It
    clean the folder as well as its direct children recursively.

    Parameters
    ----------
    path : String
        Path to the folder to be cleaned.
    """
    shutil.rmtree(path + '/__pycache__', ignore_errors=True)

    folders = glob.glob(path + '/*/')
    for folder in folders:
        shutil.rmtree(folder + '/__pycache__', ignore_errors=True)

        clean_folder(folder)

        cython_exts = glob.glob(folder + '/*.cpp')
        cython_exts.extend(glob.glob(folder + '/*.cpython*'))
        for file in cython_exts:
            os.remove(file)
