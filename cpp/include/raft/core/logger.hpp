/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include "logger-macros.hpp"

#include "logger-ext.hpp"

#if !defined(RAFT_COMPILED)
#include "logger-inl.hpp"
#endif

namespace raft {
struct log_level_setter {
  explicit log_level_setter(int level)
  {
    prev_level_ = logger::get(RAFT_NAME).get_level();
    logger::get(RAFT_NAME).set_level(level);
  }
  ~log_level_setter() { logger::get(RAFT_NAME).set_level(prev_level_); }

 private:
  int prev_level_;
};
}  // namespace raft
