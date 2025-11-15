/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace raft {

/**
 * @brief Enum for reduction/broadcast where an operation is to be performed along
 *        a matrix's rows or columns
 *
 */
enum class Apply { ALONG_ROWS, ALONG_COLUMNS };

}  // namespace raft
