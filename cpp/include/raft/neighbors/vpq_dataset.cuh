/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "dataset.hpp"
#include "detail/vpq_dataset.cuh"

#include <raft/core/resources.hpp>

namespace raft::neighbors {

/**
 * @brief Compress a dataset for use in CAGRA-Q search in place of the original data.
 *
 * @tparam DatasetT a row-major mdspan or mdarray (device or host).
 * @tparam MathT a type of the codebook elements and internal math ops.
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param[in] res
 * @param[in] params VQ and PQ parameters for compressing the data
 * @param[in] dataset a row-major mdspan or mdarray (device or host) [n_rows, dim].
 */
template <typename DatasetT,
          typename MathT = typename DatasetT::value_type,
          typename IdxT  = typename DatasetT::index_type>
auto vpq_build(const raft::resources& res, const vpq_params& params, const DatasetT& dataset)
  -> vpq_dataset<MathT, IdxT>
{
  if constexpr (std::is_same_v<MathT, half>) {
    return detail::vpq_convert_math_type<half, float, IdxT>(
      res, detail::vpq_build<DatasetT, float, IdxT>(res, params, dataset));
  } else {
    return detail::vpq_build<DatasetT, MathT, IdxT>(res, params, dataset);
  }
}

}  // namespace raft::neighbors
