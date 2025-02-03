/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <raft/core/host_mdspan.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/neighbors/detail/refine_common.hpp>
#include <raft/util/integer_utils.hpp>

#include <omp.h>

#include <algorithm>

#if defined(__arm__) || defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace raft::neighbors::detail {

// -----------------------------------------------------------------------------
//  Generic implementation
// -----------------------------------------------------------------------------

template <typename DC, typename DistanceT, typename DataT>
DistanceT euclidean_distance_squared_generic(DataT const* a, DataT const* b, size_t n) {
  // vector register capacity in elements
  size_t constexpr vreg_len = (128 / 8) / sizeof(DistanceT);
  // unroll factor = vector register capacity * number of ports;
  size_t constexpr unroll_factor = vreg_len * 4;

  // unroll factor is a power of two
  size_t n_rounded = n & (0xFFFFFFFF ^ (unroll_factor - 1));
  DistanceT distance[unroll_factor] = {0};

  for (size_t i = 0; i < n_rounded; i += unroll_factor) {
    for (size_t j = 0; j < unroll_factor; ++j) {
      distance[j] += DC::template eval<DistanceT>(a[i + j], b[i + j]);
    }
  }

  for (size_t i = n_rounded; i < n; ++i) {
    distance[i] += DC::template eval<DistanceT>(a[i], b[i]);
  }

  for (size_t i = 1; i < unroll_factor; ++i) {
    distance[0] += distance[i];
  }

  return distance[0];
}

// -----------------------------------------------------------------------------
//  NEON implementation
// -----------------------------------------------------------------------------

struct distance_comp_l2;
struct distance_comp_inner;

// fallback
template<typename DC, typename DistanceT, typename DataT>
DistanceT euclidean_distance_squared(DataT const* a, DataT const* b, size_t n) {
  return euclidean_distance_squared_generic<DC, DistanceT, DataT>(a, b, n);
}

#if defined(__arm__) || defined(__aarch64__)

template<>
inline float euclidean_distance_squared<distance_comp_l2, float, float>(
  float const* a, float const* b, size_t n) {

  int n_rounded = n - (n % 4);

  float32x4_t vreg_dsum = vdupq_n_f32(0.f);
  for (int i = 0; i < n_rounded; i += 4) {
    float32x4_t vreg_a = vld1q_f32(&a[i]);
    float32x4_t vreg_b = vld1q_f32(&b[i]);
    float32x4_t vreg_d = vsubq_f32(vreg_a, vreg_b);
    vreg_dsum = vfmaq_f32(vreg_dsum, vreg_d, vreg_d);
  }

  float dsum = vaddvq_f32(vreg_dsum);
  for (int i = n_rounded; i < n; ++i) {
      float d = a[i] - b[i];
      dsum += d * d;
  }

  return dsum;
}

template<>
inline float euclidean_distance_squared<distance_comp_l2, float, ::std::int8_t>(
  ::std::int8_t const* a, ::std::int8_t const* b, size_t n) {

  int n_rounded = n - (n % 16);
  float dsum = 0.f;

  if (n_rounded > 0) {
    float32x4_t vreg_dsum_fp32_0 = vdupq_n_f32(0.f);
    float32x4_t vreg_dsum_fp32_1 = vreg_dsum_fp32_0;
    float32x4_t vreg_dsum_fp32_2 = vreg_dsum_fp32_0;
    float32x4_t vreg_dsum_fp32_3 = vreg_dsum_fp32_0;

    for (int i = 0; i < n_rounded; i += 16) {
      int8x16_t vreg_a = vld1q_s8(&a[i]);
      int16x8_t vreg_a_s16_0 = vmovl_s8(vget_low_s8(vreg_a));
      int16x8_t vreg_a_s16_1 = vmovl_s8(vget_high_s8(vreg_a));

      int8x16_t vreg_b = vld1q_s8(&b[i]);
      int16x8_t vreg_b_s16_0 = vmovl_s8(vget_low_s8(vreg_b));
      int16x8_t vreg_b_s16_1 = vmovl_s8(vget_high_s8(vreg_b));

      int16x8_t vreg_d_s16_0 = vsubq_s16(vreg_a_s16_0, vreg_b_s16_0);
      int16x8_t vreg_d_s16_1 = vsubq_s16(vreg_a_s16_1, vreg_b_s16_1);

      float32x4_t vreg_d_fp32_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vreg_d_s16_0)));
      float32x4_t vreg_d_fp32_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vreg_d_s16_0)));
      float32x4_t vreg_d_fp32_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vreg_d_s16_1)));
      float32x4_t vreg_d_fp32_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vreg_d_s16_1)));

      vreg_dsum_fp32_0 = vfmaq_f32(vreg_dsum_fp32_0, vreg_d_fp32_0, vreg_d_fp32_0);
      vreg_dsum_fp32_1 = vfmaq_f32(vreg_dsum_fp32_1, vreg_d_fp32_1, vreg_d_fp32_1);
      vreg_dsum_fp32_2 = vfmaq_f32(vreg_dsum_fp32_2, vreg_d_fp32_2, vreg_d_fp32_2);
      vreg_dsum_fp32_3 = vfmaq_f32(vreg_dsum_fp32_3, vreg_d_fp32_3, vreg_d_fp32_3);
    }

    vreg_dsum_fp32_0 = vaddq_f32(vreg_dsum_fp32_0, vreg_dsum_fp32_1);
    vreg_dsum_fp32_2 = vaddq_f32(vreg_dsum_fp32_2, vreg_dsum_fp32_3);
    vreg_dsum_fp32_0 = vaddq_f32(vreg_dsum_fp32_0, vreg_dsum_fp32_2);

    dsum = vaddvq_f32(vreg_dsum_fp32_0); // faddp
  }

  for (int i = n_rounded; i < n; ++i) {
      float d = a[i] - b[i];
      dsum += d * d; // [nvc++] faddp, [clang] fadda, [gcc] vecsum+fadda
  }

  return dsum;
}

template<>
inline float euclidean_distance_squared<distance_comp_l2, float, ::std::uint8_t>(
  ::std::uint8_t const* a, ::std::uint8_t const* b, size_t n) {

  int n_rounded = n - (n % 16);
  float dsum = 0.f;

  if (n_rounded > 0) {
    float32x4_t vreg_dsum_fp32_0 = vdupq_n_f32(0.f);
    float32x4_t vreg_dsum_fp32_1 = vreg_dsum_fp32_0;
    float32x4_t vreg_dsum_fp32_2 = vreg_dsum_fp32_0;
    float32x4_t vreg_dsum_fp32_3 = vreg_dsum_fp32_0;

    for (int i = 0; i < n_rounded; i += 16) {
      uint8x16_t vreg_a = vld1q_u8(&a[i]);
      uint16x8_t vreg_a_u16_0 = vmovl_u8(vget_low_u8(vreg_a));
      uint16x8_t vreg_a_u16_1 = vmovl_u8(vget_high_u8(vreg_a));
      float32x4_t vreg_a_fp32_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vreg_a_u16_0)));
      float32x4_t vreg_a_fp32_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vreg_a_u16_0)));
      float32x4_t vreg_a_fp32_2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vreg_a_u16_1)));
      float32x4_t vreg_a_fp32_3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vreg_a_u16_1)));

      uint8x16_t vreg_b = vld1q_u8(&b[i]);
      uint16x8_t vreg_b_u16_0 = vmovl_u8(vget_low_u8(vreg_b));
      uint16x8_t vreg_b_u16_1 = vmovl_u8(vget_high_u8(vreg_b));
      float32x4_t vreg_b_fp32_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vreg_b_u16_0)));
      float32x4_t vreg_b_fp32_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vreg_b_u16_0)));
      float32x4_t vreg_b_fp32_2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vreg_b_u16_1)));
      float32x4_t vreg_b_fp32_3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vreg_b_u16_1)));

      float32x4_t vreg_d_fp32_0 = vsubq_f32(vreg_a_fp32_0, vreg_b_fp32_0);
      float32x4_t vreg_d_fp32_1 = vsubq_f32(vreg_a_fp32_1, vreg_b_fp32_1);
      float32x4_t vreg_d_fp32_2 = vsubq_f32(vreg_a_fp32_2, vreg_b_fp32_2);
      float32x4_t vreg_d_fp32_3 = vsubq_f32(vreg_a_fp32_3, vreg_b_fp32_3);

      vreg_dsum_fp32_0 = vfmaq_f32(vreg_dsum_fp32_0, vreg_d_fp32_0, vreg_d_fp32_0);
      vreg_dsum_fp32_1 = vfmaq_f32(vreg_dsum_fp32_1, vreg_d_fp32_1, vreg_d_fp32_1);
      vreg_dsum_fp32_2 = vfmaq_f32(vreg_dsum_fp32_2, vreg_d_fp32_2, vreg_d_fp32_2);
      vreg_dsum_fp32_3 = vfmaq_f32(vreg_dsum_fp32_3, vreg_d_fp32_3, vreg_d_fp32_3);
    }

    vreg_dsum_fp32_0 = vaddq_f32(vreg_dsum_fp32_0, vreg_dsum_fp32_1);
    vreg_dsum_fp32_2 = vaddq_f32(vreg_dsum_fp32_2, vreg_dsum_fp32_3);
    vreg_dsum_fp32_0 = vaddq_f32(vreg_dsum_fp32_0, vreg_dsum_fp32_2);

    dsum = vaddvq_f32(vreg_dsum_fp32_0); // faddp
  }

  for (int i = n_rounded; i < n; ++i) {
      float d = a[i] - b[i];
      dsum += d * d; // [nvc++] faddp, [clang] fadda, [gcc] vecsum+fadda
  }

  return dsum;
}

#endif // defined(__arm__) || defined(__aarch64__)

// -----------------------------------------------------------------------------
//  Refine kernel
// -----------------------------------------------------------------------------

template <typename DC, typename IdxT, typename DataT, typename DistanceT, typename ExtentsT>
[[gnu::optimize(3), gnu::optimize("tree-vectorize")]] void refine_host_impl(
  raft::host_matrix_view<const DataT, ExtentsT, row_major> dataset,
  raft::host_matrix_view<const DataT, ExtentsT, row_major> queries,
  raft::host_matrix_view<const IdxT, ExtentsT, row_major> neighbor_candidates,
  raft::host_matrix_view<IdxT, ExtentsT, row_major> indices,
  raft::host_matrix_view<DistanceT, ExtentsT, row_major> distances)
{
  size_t n_queries = queries.extent(0);
  size_t n_rows    = dataset.extent(0);
  size_t dim       = dataset.extent(1);
  size_t orig_k    = neighbor_candidates.extent(1);
  size_t refined_k = indices.extent(1);

  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "neighbors::refine_host(%zu, %zu -> %zu)", n_queries, orig_k, refined_k);

  auto suggested_n_threads = std::max(1, std::min(omp_get_num_procs(), omp_get_max_threads()));

  // If the number of queries is small, separate the distance calculation and
  // the top-k calculation into separate loops, and apply finer-grained thread
  // parallelism to the distance calculation loop.
  if (n_queries < size_t(suggested_n_threads)) {
    std::vector<std::vector<std::tuple<DistanceT, IdxT>>> refined_pairs(
      n_queries, std::vector<std::tuple<DistanceT, IdxT>>(orig_k));

    // For efficiency, each thread should read a certain amount of array
    // elements. The number of threads for distance computation is determined
    // taking this into account.
    auto n_elements    = std::max(size_t(512), dim);
    auto max_n_threads = raft::div_rounding_up_safe<size_t>(n_queries * orig_k * dim, n_elements);
    auto suggested_n_threads_for_distance = std::min(size_t(suggested_n_threads), max_n_threads);

    // The max number of threads for topk computation is the number of queries.
    auto suggested_n_threads_for_topk = std::min(size_t(suggested_n_threads), n_queries);

    // Compute the refined distance using original dataset vectors
#pragma omp parallel for collapse(2) num_threads(suggested_n_threads_for_distance)
    for (size_t i = 0; i < n_queries; i++) {
      for (size_t j = 0; j < orig_k; j++) {
        const DataT* query = queries.data_handle() + dim * i;
        IdxT id            = neighbor_candidates(i, j);
        DistanceT distance = 0.0;
        if (static_cast<size_t>(id) >= n_rows) {
          distance = std::numeric_limits<DistanceT>::max();
        } else {
          const DataT* row = dataset.data_handle() + dim * id;
          for (size_t k = 0; k < dim; k++) {
            distance += DC::template eval<DistanceT>(query[k], row[k]);
          }
        }
        refined_pairs[i][j] = std::make_tuple(distance, id);
      }
    }

    // Sort the query neighbors by their refined distances
#pragma omp parallel for num_threads(suggested_n_threads_for_topk)
    for (size_t i = 0; i < n_queries; i++) {
      std::sort(refined_pairs[i].begin(), refined_pairs[i].end());
      // Store first refined_k neighbors
      for (size_t j = 0; j < refined_k; j++) {
        indices(i, j) = std::get<1>(refined_pairs[i][j]);
        if (distances.data_handle() != nullptr) {
          distances(i, j) = DC::template postprocess(std::get<0>(refined_pairs[i][j]));
        }
      }
    }
    return;
  }

  if (size_t(suggested_n_threads) > n_queries) { suggested_n_threads = n_queries; }

#pragma omp parallel num_threads(suggested_n_threads)
  {
    std::vector<std::tuple<DistanceT, IdxT>> refined_pairs(orig_k);
    for (size_t i = omp_get_thread_num(); i < n_queries; i += omp_get_num_threads()) {
      // Compute the refined distance using original dataset vectors
      const DataT* query = queries.data_handle() + dim * i;
      for (size_t j = 0; j < orig_k; j++) {
        IdxT id            = neighbor_candidates(i, j);
        DistanceT distance = 0.0;
        if (static_cast<size_t>(id) >= n_rows) {
          distance = std::numeric_limits<DistanceT>::max();
        } else {
          const DataT* row = dataset.data_handle() + dim * id;
          distance = euclidean_distance_squared<DC, DistanceT, DataT>(query, row, dim);
        }
        refined_pairs[j] = std::make_tuple(distance, id);
      }
      // Sort the query neighbors by their refined distances
      std::sort(refined_pairs.begin(), refined_pairs.end());
      // Store first refined_k neighbors
      for (size_t j = 0; j < refined_k; j++) {
        indices(i, j) = std::get<1>(refined_pairs[j]);
        if (distances.data_handle() != nullptr) {
          distances(i, j) = DC::template postprocess(std::get<0>(refined_pairs[j]));
        }
      }
    }
  }
}

struct distance_comp_l2 {
  template <typename DistanceT>
  static inline auto eval(const DistanceT& a, const DistanceT& b) -> DistanceT
  {
    auto d = a - b;
    return d * d;
  }
  template <typename DistanceT>
  static inline auto postprocess(const DistanceT& a) -> DistanceT
  {
    return a;
  }
};

struct distance_comp_inner {
  template <typename DistanceT>
  static inline auto eval(const DistanceT& a, const DistanceT& b) -> DistanceT
  {
    return -a * b;
  }
  template <typename DistanceT>
  static inline auto postprocess(const DistanceT& a) -> DistanceT
  {
    return -a;
  }
};

/**
 * Naive CPU implementation of refine operation
 *
 * All pointers are expected to be accessible on the host.
 */
template <typename IdxT, typename DataT, typename DistanceT, typename ExtentsT>
[[gnu::optimize(3), gnu::optimize("tree-vectorize")]] void refine_host(
  raft::host_matrix_view<const DataT, ExtentsT, row_major> dataset,
  raft::host_matrix_view<const DataT, ExtentsT, row_major> queries,
  raft::host_matrix_view<const IdxT, ExtentsT, row_major> neighbor_candidates,
  raft::host_matrix_view<IdxT, ExtentsT, row_major> indices,
  raft::host_matrix_view<DistanceT, ExtentsT, row_major> distances,
  distance::DistanceType metric = distance::DistanceType::L2Unexpanded)
{
  refine_check_input(dataset.extents(),
                     queries.extents(),
                     neighbor_candidates.extents(),
                     indices.extents(),
                     distances.extents(),
                     metric);

  switch (metric) {
    case raft::distance::DistanceType::L2Expanded:
      return refine_host_impl<distance_comp_l2>(
        dataset, queries, neighbor_candidates, indices, distances);
    case raft::distance::DistanceType::InnerProduct:
      return refine_host_impl<distance_comp_inner>(
        dataset, queries, neighbor_candidates, indices, distances);
    default: throw raft::logic_error("Unsupported metric");
  }
}

}  // namespace raft::neighbors::detail
