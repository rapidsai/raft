#
# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

import numpy
import os
import shutil
import sys
import sysconfig
import versioneer

from distutils.sysconfig import get_python_lib
from pathlib import Path
from setuptools import find_packages
from setuptools import setup
from setuptools.extension import Extension
from setuputils import clean_folder
from setuputils import get_environment_option
from setuputils import get_cli_option

try:
    from Cython.Distutils.build_ext import new_build_ext as build_ext
except ImportError:
    from setuptools.command.build_ext import build_ext


##############################################################################
# - Dependencies include and lib folder setup --------------------------------

install_requires = [
    'cython',
]

cuda_home = get_environment_option("CUDA_HOME")

clean_artifacts = get_cli_option('clean')
single_gpu_build = get_cli_option('--singlegpu')


if not cuda_home:
    cuda_home = (
        os.popen('echo "$(dirname $(dirname $(which nvcc)))"').read().strip()
    )
    print("-- Using nvcc to detect CUDA, found at " + str(cuda_home))
cuda_include_dir = os.path.join(cuda_home, "include")
cuda_lib_dir = os.path.join(cuda_home, "lib64")

##############################################################################
# - Clean target -------------------------------------------------------------

if clean_artifacts:
    print("-- Cleaning all Python and Cython build artifacts...")

    try:
        setup_file_path = str(Path(__file__).parent.absolute())
        shutil.rmtree(setup_file_path + '/.pytest_cache', ignore_errors=True)
        shutil.rmtree(setup_file_path + '/_external_repositories',
                      ignore_errors=True)
        shutil.rmtree(setup_file_path + '/raft.egg-info', ignore_errors=True)
        shutil.rmtree(setup_file_path + '/__pycache__', ignore_errors=True)

        clean_folder(setup_file_path + '/raft')
        shutil.rmtree(setup_file_path + '/build')

    except IOError:
        pass

    # need to terminate script so cythonizing doesn't get triggered after
    # cleanup unintendedly
    sys.argv.remove("clean")

    if "--all" in sys.argv:
        sys.argv.remove("--all")

    if len(sys.argv) == 1:
        sys.exit(0)


##############################################################################
# - Cython extensions build and parameters -----------------------------------


libs = ["nccl", "cusolver", "cusparse", "cublas"]

include_dirs = [cuda_include_dir,
                numpy.get_include(),
                "../cpp/include/",
                os.path.dirname(sysconfig.get_path("include"))]

cmdclass = dict()
cmdclass.update(versioneer.get_cmdclass())
cmdclass["build_ext"] = build_ext

extensions = [
    Extension("*",
              sources=["raft/**/**/*.pyx"],
              include_dirs=include_dirs,
              library_dirs=[get_python_lib()],
              runtime_library_dirs=[cuda_lib_dir,
                                    os.path.join(os.sys.prefix, "lib")],
              libraries=libs,
              language='c++',
              extra_compile_args=['-std=c++17'])
]


##############################################################################
# - Python package generation ------------------------------------------------

setup(name='raft',
      description="RAPIDS Analytics Frameworks Toolset",
      version=versioneer.get_version(),
      classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8"
      ],
      author="NVIDIA Corporation",
      setup_requires=['cython'],
      ext_modules=extensions,
      packages=find_packages(include=['raft', 'raft.*']),
      package_data={
          "raft.common": ["*.pxd"]
      },
      install_requires=install_requires,
      license="Apache",
      cmdclass=cmdclass,
      zip_safe=False
      )
