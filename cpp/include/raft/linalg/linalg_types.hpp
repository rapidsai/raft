/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
namespace RAFT_EXPORT raft {
namespace linalg {

/**
 * @brief Enum for reduction/broadcast where an operation is to be performed along
 *        a matrix's rows or columns
 *
 */
enum class FillMode { UPPER, LOWER };

/**
 * @brief Enum for this type indicates which operation is applied to the related input (e.g. sparse
 * matrix, or vector).
 *
 */
enum class Operation { NON_TRANSPOSE, TRANSPOSE };

}  // namespace linalg
}  // namespace RAFT_EXPORT raft
