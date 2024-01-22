/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

namespace raft::linalg {

/**
 * @brief Enum for reduction/broadcast where an operation is to be performed along
 *        a matrix's rows or columns
 *
 */
enum class Apply { ALONG_ROWS, ALONG_COLUMNS };

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

}  // end namespace raft::linalg