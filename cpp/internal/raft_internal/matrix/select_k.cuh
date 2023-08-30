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

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/detail/select_radix.cuh>
#include <raft/matrix/detail/select_warpsort.cuh>
#include <raft/matrix/select_k.cuh>
#include <raft/neighbors/detail/selection_faiss.cuh>

namespace raft::matrix::select {

struct params {
  size_t batch_size;
  size_t len;
  int k;
  bool select_min;
  bool use_index_input       = true;
  bool use_same_leading_bits = false;
  bool use_memory_pool       = true;
  double frac_infinities     = 0.0;
};

inline auto operator<<(std::ostream& os, const params& ss) -> std::ostream&
{
  os << "params{batch_size: " << ss.batch_size;
  os << ", len: " << ss.len;
  os << ", k: " << ss.k;
  os << (ss.select_min ? ", asc" : ", dsc");
  if (!ss.use_index_input) { os << ", no-input-index"; }
  if (ss.use_same_leading_bits) { os << ", same-leading-bits"; }
  if (ss.frac_infinities > 0) { os << ", infs: " << ss.frac_infinities; }
  os << "}";
  return os;
}

enum class Algo {
  kPublicApi,
  kRadix8bits,
  kRadix11bits,
  kRadix11bitsExtraPass,
  kWarpAuto,
  kWarpImmediate,
  kWarpFiltered,
  kWarpDistributed,
  kWarpDistributedShm,
  kFaissBlockSelect
};

inline auto operator<<(std::ostream& os, const Algo& algo) -> std::ostream&
{
  switch (algo) {
    case Algo::kPublicApi: return os << "kPublicApi";
    case Algo::kRadix8bits: return os << "kRadix8bits";
    case Algo::kRadix11bits: return os << "kRadix11bits";
    case Algo::kRadix11bitsExtraPass: return os << "kRadix11bitsExtraPass";
    case Algo::kWarpAuto: return os << "kWarpAuto";
    case Algo::kWarpImmediate: return os << "kWarpImmediate";
    case Algo::kWarpFiltered: return os << "kWarpFiltered";
    case Algo::kWarpDistributed: return os << "kWarpDistributed";
    case Algo::kWarpDistributedShm: return os << "kWarpDistributedShm";
    case Algo::kFaissBlockSelect: return os << "kFaissBlockSelect";
    default: return os << "unknown enum value";
  }
}

template <typename T, typename IdxT>
void select_k_impl(const resources& handle,
                   const Algo& algo,
                   const T* in,
                   const IdxT* in_idx,
                   size_t batch_size,
                   size_t len,
                   int k,
                   T* out,
                   IdxT* out_idx,
                   bool select_min)
{
  auto stream = resource::get_cuda_stream(handle);
  switch (algo) {
    case Algo::kPublicApi: {
      auto in_extent  = make_extents<int64_t>(batch_size, len);
      auto out_extent = make_extents<int64_t>(batch_size, k);
      auto in_span    = make_mdspan<const T, int64_t, row_major, false, true>(in, in_extent);
      auto in_idx_span =
        make_mdspan<const IdxT, int64_t, row_major, false, true>(in_idx, in_extent);
      auto out_span     = make_mdspan<T, int64_t, row_major, false, true>(out, out_extent);
      auto out_idx_span = make_mdspan<IdxT, int64_t, row_major, false, true>(out_idx, out_extent);
      if (in_idx == nullptr) {
        // NB: std::nullopt prevents automatic inference of the template parameters.
        return matrix::select_k<T, IdxT>(
          handle, in_span, std::nullopt, out_span, out_idx_span, select_min, true);
      } else {
        return matrix::select_k(handle,
                                in_span,
                                std::make_optional(in_idx_span),
                                out_span,
                                out_idx_span,
                                select_min,
                                true);
      }
    }
    case Algo::kRadix8bits:
      return detail::select::radix::select_k<T, IdxT, 8, 512>(in,
                                                              in_idx,
                                                              batch_size,
                                                              len,
                                                              k,
                                                              out,
                                                              out_idx,
                                                              select_min,
                                                              true,  // fused_last_filter
                                                              stream);
    case Algo::kRadix11bits:
      return detail::select::radix::select_k<T, IdxT, 11, 512>(in,
                                                               in_idx,
                                                               batch_size,
                                                               len,
                                                               k,
                                                               out,
                                                               out_idx,
                                                               select_min,
                                                               true,  // fused_last_filter
                                                               stream);
    case Algo::kRadix11bitsExtraPass:
      return detail::select::radix::select_k<T, IdxT, 11, 512>(in,
                                                               in_idx,
                                                               batch_size,
                                                               len,
                                                               k,
                                                               out,
                                                               out_idx,
                                                               select_min,
                                                               false,  // fused_last_filter
                                                               stream);
    case Algo::kWarpAuto:
      return detail::select::warpsort::select_k<T, IdxT>(
        handle, in, in_idx, batch_size, len, k, out, out_idx, select_min, stream);
    case Algo::kWarpImmediate:
      return detail::select::warpsort::
        select_k_impl<T, IdxT, detail::select::warpsort::warp_sort_immediate>(
          handle, in, in_idx, batch_size, len, k, out, out_idx, select_min, stream);
    case Algo::kWarpFiltered:
      return detail::select::warpsort::
        select_k_impl<T, IdxT, detail::select::warpsort::warp_sort_filtered>(
          handle, in, in_idx, batch_size, len, k, out, out_idx, select_min, stream);
    case Algo::kWarpDistributed:
      return detail::select::warpsort::
        select_k_impl<T, IdxT, detail::select::warpsort::warp_sort_distributed>(
          handle, in, in_idx, batch_size, len, k, out, out_idx, select_min, stream);
    case Algo::kWarpDistributedShm:
      return detail::select::warpsort::
        select_k_impl<T, IdxT, detail::select::warpsort::warp_sort_distributed_ext>(
          handle, in, in_idx, batch_size, len, k, out, out_idx, select_min, stream);
    case Algo::kFaissBlockSelect:
      return neighbors::detail::select_k(
        in, in_idx, batch_size, len, out, out_idx, select_min, k, stream);
  }
}
}  // namespace raft::matrix::select
