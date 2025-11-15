/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/sparse/detail/coo.cuh>

namespace raft {
namespace sparse {

/** @brief A Container object for sparse coordinate. There are two motivations
 * behind using a container for COO arrays.
 *
 * The first motivation is that it simplifies code, rather than always having
 * to pass three arrays as function arguments.
 *
 * The second is more subtle, but much more important. The size
 * of the resulting COO from a sparse operation is often not known ahead of time,
 * since it depends on the contents of the underlying graph. The COO object can
 * allocate the underlying arrays lazily so that the object can be created by the
 * user and passed as an output argument in a sparse primitive. The sparse primitive
 * would have the responsibility for allocating and populating the output arrays,
 * while the original caller still maintains ownership of the underlying memory.
 *
 * @tparam value_t: the type of the value array.
 * @tparam value_idx: the type of index array
 *
 */
template <typename value_t, typename value_idx = int, typename nnz_t = uint64_t>
using COO = detail::COO<value_t, value_idx, nnz_t>;

};  // namespace sparse
};  // namespace raft
