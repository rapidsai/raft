/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include "kernel_sm60.cuh"
#include <cstdio>
#include <raft/distance/detail/pairwise_distance_cutlass_base.cuh>
#include <raft/linalg/contractions.cuh>
#include <utility>

namespace raft::distance::detail {

/**
 * @brief: Computes minimal alignment of row starting elements in 2D array
 *
 * The 2D matrix x is assumed to be row-major. This function computes the
 * minimal alignment in bytes of the first elements of each row.
 * Output can be 16, 8, 4, 2, 1.
 *
 * @param x        Base pointer of row-major input matrix
 * @param stride   Stride in number of element between consecutive rows.
 */
template <typename DataT>
size_t alignment_of_2d_array(const DataT* x, size_t stride)
{
  auto base           = reinterpret_cast<uintptr_t>(x);
  size_t stride_bytes = sizeof(DataT) * stride;

  for (int align = 16; align >= 0; align /= 2) {
    bool base_aligned   = base % align == 0;
    bool stride_aligned = stride_bytes % align == 0;
    if (base_aligned && stride_aligned) { return align; }
  }
  return 1;
}

template <int n>
struct alignment_tag {
  static constexpr int value = n;
};

struct alignment_dispatch {
  size_t byte_alignment = 0;

  template <typename DataT>
  alignment_dispatch(const DataT* x, const DataT* y, size_t ldx, size_t ldy)
  {
    size_t align_x = alignment_of_2d_array(x, ldx);
    size_t align_y = alignment_of_2d_array(y, ldy);

    byte_alignment = min(align_x, align_y);
  }

  template <typename F>
  auto operator()(F&& f) const
  {
    switch (byte_alignment) {
      case 16: f(alignment_tag<16>()); break;
      case 8: f(alignment_tag<8>()); break;
      case 4: f(alignment_tag<4>()); break;
      case 2: f(alignment_tag<2>()); break;
      default: f(alignment_tag<1>()); break;
    }
  }
};

template <bool rm>
struct row_major_tag {
  static constexpr int value = rm;
};

struct row_major_dispatch {
  bool is_row_major_;
  row_major_dispatch(bool row_major) : is_row_major_(row_major) {}

  template <typename F>
  auto operator()(F&& f) const
  {
    if (is_row_major_) {
      f(row_major_tag<true>());
    } else {
      f(row_major_tag<false>());
    }
  }
};

template <typename F1, typename F2>
auto join_dispatch(F1&& f1, F2&& f2)
{
  const auto lam = [f1, f2](auto f) {
    f1([f, f2](auto... args1) { f2([f, args1...](auto... args2) { f(args1..., args2...); }); });
  };
  return lam;
}

template <typename F1, typename F2, typename... Fs>
auto join_dispatch(F1 f1, F2 f2, Fs... fs)
{
  return join_dispatch(join_dispatch(f1, f2), std::forward<Fs>(fs)...);
}

template <typename opT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename FinOpT,
          typename IdxT = int>
void distance_matrix_dispatch(opT distance_op,
                              IdxT m,
                              IdxT n,
                              IdxT k,
                              const DataT* x,
                              const DataT* y,
                              const DataT* x_norm,
                              const DataT* y_norm,
                              OutT* out,
                              FinOpT fin_op,
                              cudaStream_t stream,
                              bool is_row_major)
{
  // Determine leading dimensions and possibly flip order of passing x and y if
  // column_major.
  IdxT ldx, ldy, ld_out;
  if (is_row_major) {
    ldx = k, ldy = k, ld_out = n;
  } else {
    // Flip x, y, and m, n.
    std::swap<const DataT*>(x, y);
    std::swap<const DataT*>(x_norm, y_norm);
    std::swap(m, n);
    ldx = m, ldy = n, ld_out = n;
  }

  alignment_dispatch d_align(x, y, ldx, ldy);
  row_major_dispatch d_row_major(is_row_major);
  auto dispatch = join_dispatch(d_align, d_row_major);

  dispatch([&](auto alignment_tag, auto row_major_tag) {
    // Compute number of elements that can be loaded in one instruction
    // without causing misalignent errors.
    constexpr int vec_len_ideal =
      (alignment_tag.value % sizeof(DataT) == 0) ? alignment_tag.value / sizeof(DataT) : 1;

    // To keep compile times in check, we only specialize on veclen > 1 when
    // the inner loop is relatively cheap (< 5 flops).
    constexpr int vec_len = distance_op.expensive_inner_loop ? 1 : vec_len_ideal;

    typedef typename raft::linalg::Policy4x4<DataT, vec_len>::Policy RowPolicy;
    typedef typename raft::linalg::Policy4x4<DataT, vec_len>::ColPolicy ColPolicy;
    typedef typename std::conditional<row_major_tag.value, RowPolicy, ColPolicy>::type Policy;

    // Create compile-time template parameter
    using KP_T = kernel_params_T<DataT, AccT, OutT, IdxT, Policy, opT, FinOpT, row_major_tag.value>;

    return pairwise_matrix<KP_T>(
      distance_op, fin_op, x, y, x_norm, y_norm, m, n, k, ldx, ldy, ld_out, out, stream);
  });
}

template <typename opT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename FinOpT,
          typename IdxT = int>
void distance_matrix_cutlass_dispatch(opT cutlass_op,
                                      IdxT m,
                                      IdxT n,
                                      IdxT k,
                                      const DataT* x,
                                      const DataT* y,
                                      const DataT* x_norm,
                                      const DataT* y_norm,
                                      OutT* out,
                                      FinOpT fin_op,
                                      cudaStream_t stream,
                                      bool is_row_major)
{
  // Determine leading dimensions and possibly flip order of passing x and y if
  // column_major.
  IdxT ldx, ldy, ld_out;
  if (is_row_major) {
    ldx = k, ldy = k, ld_out = n;
  } else {
    std::swap<const DataT*>(x, y);
    std::swap<const DataT*>(x_norm, y_norm);
    std::swap(m, n);
    ldx = m, ldy = n, ld_out = n;
  }

  alignment_dispatch d_align(x, y, ldx, ldy);
  row_major_dispatch d_row_major(is_row_major);

  auto dispatch = join_dispatch(d_align, d_row_major);

  dispatch([&](auto alignment_tag, auto row_major_tag) {
    constexpr int vec_len =
      (alignment_tag.value % sizeof(DataT) == 0) ? alignment_tag.value / sizeof(DataT) : 1;

    cutlassDistanceKernel<DataT, AccT, OutT, IdxT, vec_len, FinOpT, opT, row_major_tag.value>(
      x, y, x_norm, y_norm, m, n, k, ldx, ldy, ld_out, out, fin_op, cutlass_op, stream);
  });
}

};  // namespace raft::distance::detail
