/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "util.h"

#include <sys/stat.h>
#include <sys/types.h>

#include <cstring>
#include <sstream>

namespace raft::bench::ann {

std::vector<std::string> split(const std::string& s, char delimiter)
{
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream iss(s);
  while (getline(iss, token, delimiter)) {
    if (!token.empty()) { tokens.push_back(token); }
  }
  return tokens;
}

bool file_exists(const std::string& filename)
{
  struct stat statbuf;
  if (stat(filename.c_str(), &statbuf) != 0) { return false; }
  return S_ISREG(statbuf.st_mode);
}

bool dir_exists(const std::string& dir)
{
  struct stat statbuf;
  if (stat(dir.c_str(), &statbuf) != 0) { return false; }
  return S_ISDIR(statbuf.st_mode);
}

bool create_dir(const std::string& dir)
{
  const auto path = split(dir, '/');

  std::string cwd;
  if (!dir.empty() && dir[0] == '/') { cwd += '/'; }

  for (const auto& p : path) {
    cwd += p + "/";
    if (!dir_exists(cwd)) {
      int ret = mkdir(cwd.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
      if (ret != 0) { return false; }
    }
  }
  return true;
}

}  // namespace raft::bench::ann
