/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#ifndef __RAFT_RT_MACRO_UTILS_H
#define __RAFT_RT_MACRO_UTILS_H

#pragma once

/**
 * Some macro magic to remove optional parentheses of a macro argument.
 * See https://stackoverflow.com/a/62984543
 */
#define RAFT_DEPAREN(X)      RAFT_DEPAREN_H2(RAFT_DEPAREN_H1 X)
#define RAFT_DEPAREN_H1(...) RAFT_DEPAREN_H1 __VA_ARGS__
#define RAFT_DEPAREN_H2(...) RAFT_DEPAREN_H3(__VA_ARGS__)
#define RAFT_DEPAREN_H3(...) RAFT_DEPAREN_MAGIC##__VA_ARGS__
#define RAFT_DEPAREN_MAGICRAFT_DEPAREN_H1

#define RAFT_STRINGIFY_DETAIL(...) #__VA_ARGS__
#define RAFT_STRINGIFY(...)        RAFT_STRINGIFY_DETAIL(__VA_ARGS__)

#endif