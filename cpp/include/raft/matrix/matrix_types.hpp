/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
namespace RAFT_EXPORT raft {
namespace matrix {

struct print_separators {
  char horizontal = ' ';
  char vertical   = '\n';
};

}  // namespace matrix
}  // namespace RAFT_EXPORT raft
