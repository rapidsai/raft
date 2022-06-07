#!/usr/bin/env python3

import re
import sys
import os
from os.path import dirname, join as path_join, abspath, exists

extra_paths = [path_join(dirname(abspath(__file__)), "include")]

def find_file(included_name, current_file):
    current_dir = dirname(abspath(current_file))
    for idir in [current_dir] + extra_paths:
        try_path = path_join(idir, included_name)
        if exists(try_path):
            return try_path
    return None

def process_file(file_path, out_lines=[], front_matter_lines=[], back_matter_lines=[], processed_files=[]):
    out_lines += "//BEGIN_FILE_INCLUDE: " + file_path + "\n"
    with open(file_path, "r") as f:
        for line in f:
            m_inc = re.match(r'# *include\s*[<"](.+)[>"]\s*', line)
            if m_inc:
                inc_name = m_inc.group(1)
                inc_path = find_file(inc_name, file_path)
                if inc_path not in processed_files:
                    if inc_path is not None:
                        processed_files += [inc_path]
                        process_file(inc_path, out_lines, front_matter_lines, back_matter_lines, processed_files)
                    else:
                        # assume it's a system header; add it to the front matter just to be clean
                        front_matter_lines += [line]
                continue
            m_once = re.match(r"#pragma once\s*", line)
            # ignore pragma once; we're handling it here
            if m_once:
                continue
            # otherwise, just add the line to the output
            if line[-1] != "\n": line = line + "\n"
            out_lines += [line]
    out_lines += "//END_FILE_INCLUDE: " + file_path + "\n"
    return "".join(front_matter_lines) + "\n" + "".join(out_lines) + "".join(back_matter_lines)

if __name__ == "__main__":
    print(process_file(abspath(sys.argv[1]), [],
      # We use an include guard instead of `#pragma once` because Godbolt will
      # cause complaints about `#pragma once` when they are used in URL includes.
      ["#ifndef _MDSPAN_SINGLE_HEADER_INCLUDE_GUARD_\n",
       "#define _MDSPAN_SINGLE_HEADER_INCLUDE_GUARD_\n"],
      ["#endif // _MDSPAN_SINGLE_HEADER_INCLUDE_GUARD_\n"],
      [abspath(sys.argv[1])]))

