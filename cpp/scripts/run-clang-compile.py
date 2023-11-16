# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# IMPORTANT DISCLAIMER:                                                       #
# This file is experimental and may not run successfully on the entire repo!  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#

from __future__ import print_function
import argparse
import json
import multiprocessing as mp
import os
import re
import shutil
import subprocess


CMAKE_COMPILER_REGEX = re.compile(
    r"^\s*CMAKE_CXX_COMPILER:FILEPATH=(.+)\s*$", re.MULTILINE)
CLANG_COMPILER = "clang++"
GPU_ARCH_REGEX = re.compile(r"sm_(\d+)")
SPACES = re.compile(r"\s+")
XCOMPILER_FLAG = re.compile(r"-((Xcompiler)|(-compiler-options))=?")
XPTXAS_FLAG = re.compile(r"-((Xptxas)|(-ptxas-options))=?")
# any options that may have equal signs in nvcc but not in clang
# add those options here if you find any
OPTIONS_NO_EQUAL_SIGN = ['-isystem']
SEPARATOR = "-" * 8
END_SEPARATOR = "*" * 64


def parse_args():
    argparser = argparse.ArgumentParser("Runs clang++ on a project instead of nvcc")
    argparser.add_argument(
        "-cdb", type=str, default="compile_commands.json",
        help="Path to cmake-generated compilation database")
    argparser.add_argument(
        "-ignore", type=str, default=None,
        help="Regex used to ignore files from checking")
    argparser.add_argument(
        "-select", type=str, default=None,
        help="Regex used to select files for checking")
    argparser.add_argument(
        "-j", type=int, default=-1, help="Number of parallel jobs to launch.")
    argparser.add_argument(
        "-build_dir", type=str, default=None,
        help="Directory from which compile commands should be called. "
        "By default, directory of compile_commands.json file.")
    args = argparser.parse_args()
    if args.j <= 0:
        args.j = mp.cpu_count()
    args.ignore_compiled = re.compile(args.ignore) if args.ignore else None
    args.select_compiled = re.compile(args.select) if args.select else None
    # we don't check clang's version, it should be OK with any clang
    # recent enough to handle CUDA >= 11
    if not os.path.exists(args.cdb):
        raise Exception("Compilation database '%s' missing" % args.cdb)
    if args.build_dir is None:
        args.build_dir = os.path.dirname(args.cdb)
    return args


def get_gcc_root(build_dir):
    # first try to determine GCC based on CMakeCache
    cmake_cache = os.path.join(build_dir, "CMakeCache.txt")
    if os.path.isfile(cmake_cache):
        with open(cmake_cache) as f:
            content = f.read()
        match = CMAKE_COMPILER_REGEX.search(content)
        if match:
            return os.path.dirname(os.path.dirname(match.group(1)))
    # first fall-back to CONDA prefix if we have a build sysroot there
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    conda_sysroot = os.environ.get("CONDA_BUILD_SYSROOT", "")
    if conda_prefix and conda_sysroot:
        return conda_prefix
    # second fall-back to default g++ install
    default_gxx = shutil.which("g++")
    if default_gxx:
        return os.path.dirname(os.path.dirname(default_gxx))
    raise Exception("Cannot find any g++ install on the system.")


def list_all_cmds(cdb):
    with open(cdb, "r") as fp:
        return json.load(fp)


def get_gpu_archs(command):
    # clang only accepts a single architecture, so first determine the lowest
    archs = []
    for loc in range(len(command)):
        if (command[loc] != "-gencode" and command[loc] != "--generate-code"
                and not command[loc].startswith("--generate-code=")):
            continue
        if command[loc].startswith("--generate-code="):
            arch_flag = command[loc][len("--generate-code="):]
        else:
            arch_flag = command[loc + 1]
        match = GPU_ARCH_REGEX.search(arch_flag)
        if match is not None:
            archs.append(int(match.group(1)))
    return ["--cuda-gpu-arch=sm_%d" % min(archs)]


def get_index(arr, item_options):
    return set(i for i, s in enumerate(arr) for item in item_options
               if s == item)


def remove_items(arr, item_options):
    for i in sorted(get_index(arr, item_options), reverse=True):
        del arr[i]


def remove_items_plus_one(arr, item_options):
    for i in sorted(get_index(arr, item_options), reverse=True):
        if i < len(arr) - 1:
            del arr[i + 1]
        del arr[i]
    idx = set(i for i, s in enumerate(arr) for item in item_options
              if s.startswith(item + "="))
    for i in sorted(idx, reverse=True):
        del arr[i]


def add_cuda_path(command, nvcc):
    nvcc_path = shutil.which(nvcc)
    if not nvcc_path:
        raise Exception("Command %s has invalid compiler %s" % (command, nvcc))
    cuda_root = os.path.dirname(os.path.dirname(nvcc_path))
    command.append('--cuda-path=%s' % cuda_root)


def get_clang_args(cmd, build_dir):
    command, file = cmd["command"], cmd["file"]
    is_cuda = file.endswith(".cu")
    command = re.split(SPACES, command)
    # get original compiler
    cc_orig = command[0]
    # compiler is always clang++!
    command[0] = "clang++"
    # remove compilation and output targets from the original command
    remove_items_plus_one(command, ["--compile", "-c"])
    remove_items_plus_one(command, ["--output-file", "-o"])
    if is_cuda:
        # replace nvcc's "-gencode ..." with clang's "--cuda-gpu-arch ..."
        archs = get_gpu_archs(command)
        command.extend(archs)
        # provide proper cuda path to clang
        add_cuda_path(command, cc_orig)
        # remove all kinds of nvcc flags clang doesn't know about
        remove_items_plus_one(command, [
            "--generate-code",
            "-gencode",
            "--x",
            "-x",
            "--compiler-bindir",
            "-ccbin",
            "--diag_suppress",
            "-diag-suppress",
            "--default-stream",
            "-default-stream",
        ])
        remove_items(command, [
            "-extended-lambda",
            "--extended-lambda",
            "-expt-extended-lambda",
            "--expt-extended-lambda",
            "-expt-relaxed-constexpr",
            "--expt-relaxed-constexpr",
            "--device-debug",
            "-G",
            "--generate-line-info",
            "-lineinfo",
        ])
        # "-x cuda" is the right usage in clang
        command.extend(["-x", "cuda"])
        # we remove -Xcompiler flags: here we basically have to hope for the
        # best that clang++ will accept any flags which nvcc passed to gcc
        for i, c in reversed(list(enumerate(command))):
            new_c = XCOMPILER_FLAG.sub('', c)
            if new_c == c:
                continue
            command[i:i + 1] = new_c.split(',')
        # we also change -Xptxas to -Xcuda-ptxas, always adding space here
        for i, c in reversed(list(enumerate(command))):
            if XPTXAS_FLAG.search(c):
                if not c.endswith("=") and i < len(command) - 1:
                    del command[i + 1]
                command[i] = '-Xcuda-ptxas'
                command.insert(i + 1, XPTXAS_FLAG.sub('', c))
        # several options like isystem don't expect `=`
        for opt in OPTIONS_NO_EQUAL_SIGN:
            opt_eq = opt + '='
            # make sure that we iterate from back to front here for insert
            for i, c in reversed(list(enumerate(command))):
                if not c.startswith(opt_eq):
                    continue
                x = c.split('=')
                # we only care about the first `=`
                command[i] = x[0]
                command.insert(i + 1, '='.join(x[1:]))
        # use extensible whole program, to avoid ptx resolution/linking
        command.extend(["-Xcuda-ptxas", "-ewp"])
        # for libcudacxx, we need to allow variadic functions
        command.extend(["-Xclang", "-fcuda-allow-variadic-functions"])
        # add some additional CUDA intrinsics
        cuda_intrinsics_file = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "__clang_cuda_additional_intrinsics.h")
        command.extend(["-include", cuda_intrinsics_file])
    # somehow this option gets onto the commandline, it is unrecognized by clang
    remove_items(command, [
        "--forward-unknown-to-host-compiler",
        "-forward-unknown-to-host-compiler"
    ])
    # do not treat warnings as errors here !
    for i, x in reversed(list(enumerate(command))):
        if x.startswith("-Werror"):
            del command[i]
    # try to figure out which GCC CMAKE used, and tell clang all about it
    command.append("--gcc-toolchain=%s" % get_gcc_root(build_dir))
    return command


def run_clang_command(clang_cmd, cwd):
    cmd = " ".join(clang_cmd)
    result = subprocess.run(cmd, check=False, shell=True, cwd=cwd,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    result.stdout = result.stdout.decode("utf-8").strip()
    out = "CMD: " + cmd + "\n"
    out += "CWD: " + cwd + "\n"
    out += "EXIT-CODE: %d\n" % result.returncode
    status = result.returncode == 0
    out += result.stdout
    return status, out


class LockContext(object):
    def __init__(self, lock=None) -> None:
        self._lock = lock
    
    def __enter__(self):
        if self._lock:
            self._lock.acquire()
        return self
    
    def __exit__(self, _, __, ___):
        if self._lock:
            self._lock.release()
        return False  # we don't handle exceptions


def print_result(passed, stdout, file):
    status_str = "PASSED" if passed else "FAILED"
    print("%s File:%s %s %s" % (SEPARATOR, file, status_str, SEPARATOR))
    if not passed and stdout:
        print(stdout)
        print("%s\n" % END_SEPARATOR)


def run_clang(cmd, args):
    command = get_clang_args(cmd, args.build_dir)
    # compile only and dump output to /dev/null
    command.extend(["-c", cmd["file"], "-o", os.devnull])
    status, out = run_clang_command(command, args.build_dir)
    # we immediately print the result since this is more interactive for user
    with lock:
        print_result(status, out, cmd["file"])
        return status


# mostly used for debugging purposes
def run_sequential(args, all_files):
    # lock must be defined as in `run_parallel`
    global lock
    lock = LockContext()
    results = []
    for cmd in all_files:
        # skip files that we don't want to look at
        if args.ignore_compiled is not None and \
           re.search(args.ignore_compiled, cmd["file"]) is not None:
            continue
        if args.select_compiled is not None and \
           re.search(args.select_compiled, cmd["file"]) is None:
            continue
        results.append(run_clang(cmd, args))
    return all(results)


def copy_lock(init_lock):
    # this is required to pass locks to pool workers
    # see https://stackoverflow.com/questions/25557686/
    # python-sharing-a-lock-between-processes
    global lock
    lock = init_lock


def run_parallel(args, all_files):
    init_lock = LockContext(mp.Lock())
    pool = mp.Pool(args.j, initializer=copy_lock, initargs=(init_lock,))
    results = []
    for cmd in all_files:
        # skip files that we don't want to look at
        if args.ignore_compiled is not None and \
           re.search(args.ignore_compiled, cmd["file"]) is not None:
            continue
        if args.select_compiled is not None and \
           re.search(args.select_compiled, cmd["file"]) is None:
            continue
        results.append(pool.apply_async(run_clang, args=(cmd, args)))
    results_final = [r.get() for r in results]
    pool.close()
    pool.join()
    return all(results_final)


def main():
    args = parse_args()
    all_files = list_all_cmds(args.cdb)
    # ensure that we use only the real paths
    for cmd in all_files:
        cmd["file"] = os.path.realpath(os.path.expanduser(cmd["file"]))
    if args.j == 1:
        status = run_sequential(args, all_files)
    else:
        status = run_parallel(args, all_files)
    if not status:
        raise Exception("clang++ failed! Refer to the errors above.")


if __name__ == "__main__":
    main()
