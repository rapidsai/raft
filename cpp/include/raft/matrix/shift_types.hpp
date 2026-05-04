/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
namespace RAFT_EXPORT raft {
namespace matrix {

enum ShiftDirection { TOWARDS_BEGINNING, TOWARDS_END };
enum ShiftType { ROW, COL };

}  // namespace matrix
}  // namespace RAFT_EXPORT raft
