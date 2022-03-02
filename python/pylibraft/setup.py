#
# Copyright (c) 2022, NVIDIA CORPORATION.
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

# Must import in this order:
#   setuptools -> Cython.Distutils.build_ext -> setuptools.command.build_ext
# Otherwise, setuptools.command.build_ext ends up inheriting from
# Cython.Distutils.old_build_ext which we do not want
import setuptools

try:
    from Cython.Distutils.build_ext import new_build_ext as _build_ext
except ImportError:
    from setuptools.command.build_ext import build_ext as _build_ext

from distutils.sysconfig import get_python_lib

import setuptools.command.build_ext
from setuptools import find_packages, setup
from setuptools.extension import Extension

from setuputils import clean_folder
from setuputils import get_environment_option
from setuputils import get_cli_option

from pathlib import Path

import versioneer


##############################################################################
# - Dependencies include and lib folder setup --------------------------------

install_requires = [
    'cython'
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
        shutil.rmtree(setup_file_path + '/pylibraft.egg-info',
                      ignore_errors=True)
        shutil.rmtree(setup_file_path + '/__pycache__', ignore_errors=True)

        clean_folder(setup_file_path + '/pylibraft')
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

libs = ['raft_distance', 'cudart', "cusolver", "cusparse", "cublas"]

include_dirs = [cuda_include_dir,
                numpy.get_include(),
                "../cpp/include/",
                os.path.dirname(sysconfig.get_path("include"))]

extensions = [
    Extension("*",
              sources=["pylibraft/**/*.pyx"],
              include_dirs=include_dirs,
              library_dirs=[get_python_lib()],
              runtime_library_dirs=[cuda_lib_dir,
                                    os.path.join(os.sys.prefix, "lib")],
              libraries=libs,
              language='c++',
              extra_compile_args=['-std=c++17'])
]


class build_ext_no_debug(_build_ext):

    def build_extensions(self):
        def remove_flags(compiler, *flags):
            for flag in flags:
                try:
                    compiler.compiler_so = list(
                        filter((flag).__ne__, compiler.compiler_so)
                    )
                except Exception:
                    pass

        # Full optimization
        self.compiler.compiler_so.append("-O3")

        # Ignore deprecation declaration warnings
        self.compiler.compiler_so.append("-Wno-deprecated-declarations")

        # No debug symbols, full optimization, no '-Wstrict-prototypes' warning
        remove_flags(
            self.compiler, "-g", "-G", "-O1", "-O2", "-Wstrict-prototypes"
        )
        super().build_extensions()

    def finalize_options(self):
        if self.distribution.ext_modules:
            # Delay import this to allow for Cython-less installs
            from Cython.Build.Dependencies import cythonize

            nthreads = getattr(self, "parallel", None)  # -j option in Py3.5+
            nthreads = int(nthreads) if nthreads else None
            self.distribution.ext_modules = cythonize(
                self.distribution.ext_modules,
                nthreads=nthreads,
                force=self.force,
                gdb_debug=False,
                compiler_directives=dict(
                    profile=False, language_level=3, embedsignature=True
                ),
            )
        # Skip calling super() and jump straight to setuptools
        setuptools.command.build_ext.build_ext.finalize_options(self)


cmdclass = dict()
cmdclass.update(versioneer.get_cmdclass())
cmdclass["build_ext"] = build_ext_no_debug


##############################################################################
# - Python package generation ------------------------------------------------


setup(name='pylibraft',
      description="RAFT: Reusable Algorithms Functions and other Tools",
      version=versioneer.get_version(),
      classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7"
      ],
      author="NVIDIA Corporation",
      setup_requires=['cython'],
      ext_modules=extensions,
      package_data=dict.fromkeys(
                         find_packages(include=["pylibraft.distance",
                                                "pylibraft.distance.includes",
                                                "pylibraft.common",
                                                "pylibraft.common.includes"]),
                         ["*.hpp", "*.pxd"],
      ),
      packages=find_packages(include=['pylibraft', 'pylibraft.*']),
      install_requires=install_requires,
      license="Apache",
      cmdclass=cmdclass,
      zip_safe=False
      )
