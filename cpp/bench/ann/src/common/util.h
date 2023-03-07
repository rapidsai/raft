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
#pragma once

#include <chrono>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

namespace raft::bench::ann {

class Timer {
 public:
  Timer() { reset(); }
  void reset() { start_time_ = std::chrono::steady_clock::now(); }
  float elapsed_ms()
  {
    auto end_time = std::chrono::steady_clock::now();
    auto dur =
      std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(end_time - start_time_);
    return dur.count();
  }

 private:
  std::chrono::steady_clock::time_point start_time_;
};

std::vector<std::string> split(const std::string& s, char delimiter);

bool file_exists(const std::string& filename);
bool dir_exists(const std::string& dir);
bool create_dir(const std::string& dir);

template <typename... Ts>
void log_(const char* level, Ts... vs)
{
  char buf[20];
  std::time_t now = std::time(nullptr);
  std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
  printf("%s [%s] ", buf, level);
  printf(vs...);
  printf("\n");
  fflush(stdout);
}

template <typename... Ts>
void log_info(Ts... vs)
{
  log_("info", vs...);
}

template <typename... Ts>
void log_warn(Ts... vs)
{
  log_("warn", vs...);
}

template <typename... Ts>
void log_error(Ts... vs)
{
  log_("error", vs...);
}

}  // namespace raft::bench::ann
