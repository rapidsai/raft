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

#include <sstream>

#if (RAFT_LOG_ACTIVE_LEVEL <= RAFT_LOG_LEVEL_TRACE)
#define RAFT_LOG_TRACE_VEC(ptr, len)                                               \
  do {                                                                             \
    std::stringstream ss;                                                          \
    ss << raft::detail::format("%s:%d ", __FILE__, __LINE__);                      \
    print_vector(#ptr, ptr, len, ss);                                              \
    raft::logger::get(RAFT_NAME).log(RAFT_LEVEL_TRACE, ss.str().c_str());          \
    RAFT_LOGGER_CALL(raft::default_logger(), raft::level_enum::trace, __VA_ARGS__) \
  } while (0)
#else
#define RAFT_LOG_TRACE_VEC(ptr, len) void(0)
#endif
