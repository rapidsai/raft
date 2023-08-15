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

#include <raft/core/cublas_macros.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/resource/cublaslt_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cache.hpp>
#include <raft/util/cuda_data_type.hpp>

#include <cuda_fp16.hpp>
#include <type_traits>

namespace raft::linalg::detail {

/** Get the cublas compute type for the combination of input types. */
template <typename S, typename A, typename B, typename C>
auto get_matmul_type() -> cublasComputeType_t
{
  static_assert(std::is_same_v<S, float> && std::is_same_v<A, float> && std::is_same_v<B, float> &&
                  std::is_same_v<C, float>,
                "Unsupported combination of input types. Consult cublas API for supported types.");
  return CUBLAS_COMPUTE_32F;
}

template <>
inline auto get_matmul_type<float, float, float, float>() -> cublasComputeType_t
{
  return CUBLAS_COMPUTE_32F;
}
template <>
inline auto get_matmul_type<float, half, half, float>() -> cublasComputeType_t
{
  return CUBLAS_COMPUTE_32F;
}
template <>
inline auto get_matmul_type<float, int8_t, int8_t, float>() -> cublasComputeType_t
{
  return CUBLAS_COMPUTE_32F;
}
template <>
inline auto get_matmul_type<float, half, half, half>() -> cublasComputeType_t
{
  return CUBLAS_COMPUTE_32F;
}
template <>
inline auto get_matmul_type<half, half, half, half>() -> cublasComputeType_t
{
  return CUBLAS_COMPUTE_16F;
}
template <>
inline auto get_matmul_type<int32_t, int8_t, int8_t, int32_t>() -> cublasComputeType_t
{
  return CUBLAS_COMPUTE_32I;
}
template <>
inline auto get_matmul_type<float, int8_t, int8_t, int8_t>() -> cublasComputeType_t
{
  return CUBLAS_COMPUTE_32I;
}
template <>
inline auto get_matmul_type<double, double, double, double>() -> cublasComputeType_t
{
  return CUBLAS_COMPUTE_64F;
}

/** Unique representation of a matrix multiplication (assuming fixed types). */
struct matmul_key_t {
  uint64_t m;
  uint64_t n;
  uint64_t k;
  uint64_t lda;
  uint64_t ldb;
  uint64_t ldc;
  bool trans_a;
  bool trans_b;
};

inline auto operator==(const matmul_key_t& a, const matmul_key_t& b) -> bool
{
  return a.m == b.m && a.n == b.n && a.k == b.k && a.lda == b.lda && a.ldb == b.ldb &&
         a.ldc == b.ldc && a.trans_a == b.trans_a && a.trans_b == b.trans_b;
}

struct matmul_key_hash {
  inline auto operator()(const matmul_key_t& x) const noexcept -> std::size_t
  {
    return x.m * x.n * x.k + x.lda * x.ldb * x.ldc + size_t{x.trans_a} + size_t{x.trans_b} * 2;
  }
};

/** Descriptor for a column-major cublasLt matrix. */
struct cublastlt_matrix_layout {
  cublasLtMatrixLayout_t res{nullptr};
  inline cublastlt_matrix_layout(cudaDataType dtype, uint64_t rows, uint64_t cols, uint64_t ld)
  {
    RAFT_CUBLAS_TRY(cublasLtMatrixLayoutCreate(&res, dtype, rows, cols, ld));
  }
  inline cublastlt_matrix_layout(const cublastlt_matrix_layout&)                    = delete;
  inline auto operator=(const cublastlt_matrix_layout&) -> cublastlt_matrix_layout& = delete;
  inline cublastlt_matrix_layout(cublastlt_matrix_layout&&)                         = default;
  inline auto operator=(cublastlt_matrix_layout&&) -> cublastlt_matrix_layout&      = default;

  inline ~cublastlt_matrix_layout() noexcept
  {
    RAFT_CUBLAS_TRY_NO_THROW(cublasLtMatrixLayoutDestroy(res));
  }

  // NOLINTNEXTLINE
  inline operator cublasLtMatrixLayout_t() const noexcept { return res; }

  template <typename T>
  static inline auto for_matmul(bool col_major, uint64_t rows, uint64_t cols, uint64_t ld)
    -> cublastlt_matrix_layout
  {
    return cublastlt_matrix_layout{
      get_cuda_data_type<T>(), col_major ? rows : cols, col_major ? cols : rows, ld};
  }
};

/** Descriptor for a cublasLt matmul function. */
struct cublastlt_matmul_desc {
  cublasLtMatmulDesc_t res{nullptr};
  inline cublastlt_matmul_desc(cublasComputeType_t compute_type, cudaDataType scale_type)
  {
    RAFT_CUBLAS_TRY(cublasLtMatmulDescCreate(&res, compute_type, scale_type));
  }
  inline cublastlt_matmul_desc(const cublastlt_matmul_desc&)                    = delete;
  inline auto operator=(const cublastlt_matmul_desc&) -> cublastlt_matmul_desc& = delete;
  inline cublastlt_matmul_desc(cublastlt_matmul_desc&&)                         = default;
  inline auto operator=(cublastlt_matmul_desc&&) -> cublastlt_matmul_desc&      = default;

  inline ~cublastlt_matmul_desc() noexcept
  {
    RAFT_CUBLAS_TRY_NO_THROW(cublasLtMatmulDescDestroy(res));
  }

  // NOLINTNEXTLINE
  inline operator cublasLtMatmulDesc_t() const noexcept { return res; }

  template <typename S, typename A, typename B, typename C, bool DevicePointerMode = false>
  static inline auto for_matmult(bool transpose_a, bool transpose_b) -> cublastlt_matmul_desc
  {
    auto desc = cublastlt_matmul_desc{get_matmul_type<S, A, B, C>(), get_cuda_data_type<S>()};
    if constexpr (DevicePointerMode) {
      const cublasPointerMode_t mode = CUBLAS_POINTER_MODE_DEVICE;
      RAFT_CUBLAS_TRY(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &mode, sizeof(mode)));
    }
    const cublasOperation_t trans_op = CUBLAS_OP_T;
    if (transpose_a) {
      RAFT_CUBLAS_TRY(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_op, sizeof(trans_op)));
    }
    if (transpose_b) {
      RAFT_CUBLAS_TRY(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_op, sizeof(trans_op)));
    }
    return desc;
  }
};

/** Full description of matmul. */
struct matmul_desc {
  cublastlt_matmul_desc desc;
  cublastlt_matrix_layout a;
  cublastlt_matrix_layout b;
  cublastlt_matrix_layout c;
  cublasLtMatmulHeuristicResult_t heuristics;

  template <typename S, typename A, typename B, typename C, bool DevicePointerMode = false>
  static inline auto create(raft::resources const& res, const matmul_key_t& args) -> matmul_desc
  {
    matmul_desc r{
      cublastlt_matmul_desc::for_matmult<S, A, B, C, DevicePointerMode>(args.trans_a, args.trans_b),
      cublastlt_matrix_layout::for_matmul<A>(!(args.trans_a), args.m, args.k, args.lda),
      cublastlt_matrix_layout::for_matmul<B>(!(args.trans_b), args.k, args.n, args.ldb),
      cublastlt_matrix_layout::for_matmul<C>(true, args.m, args.n, args.ldc)};
    int algo_count;
    cublasLtMatmulPreference_t preference;
    RAFT_CUBLAS_TRY(cublasLtMatmulPreferenceCreate(&preference));
    RAFT_CUBLAS_TRY(cublasLtMatmulAlgoGetHeuristic(resource::get_cublaslt_handle(res),
                                                   r.desc,
                                                   r.a,
                                                   r.b,
                                                   r.c,
                                                   r.c,
                                                   preference,
                                                   1,
                                                   &r.heuristics,
                                                   &algo_count));
    RAFT_CUBLAS_TRY(cublasLtMatmulPreferenceDestroy(preference));
    return r;
  }
};

/** Number of matmul invocations to cache. */
static constexpr size_t kLRUSize = 100;

/**
 * @brief the wrapper of cublasLt matmul function
 *  It computes the following equation: C = alpha .* opA(A) * opB(B) + beta .* C
 *
 * @tparam DevicePointerMode whether pointers alpha, beta point to device memory
 * @tparam S the type of scale parameters alpha, beta
 * @tparam A the element type of matrix A
 * @tparam B the element type of matrix B
 * @tparam C the element type of matrix C
 *
 * @param [in] res raft resources
 * @param [in] trans_a cublas transpose op for A
 * @param [in] trans_b cublas transpose op for B
 * @param [in] m number of rows of C
 * @param [in] n number of columns of C
 * @param [in] k number of rows of opB(B) / number of columns of opA(A)
 * @param [in] alpha host or device scalar
 * @param [in] a_ptr such a matrix that the shape of column-major opA(A) is [m, k]
 * @param [in] lda leading dimension of A
 * @param [in] b_ptr such a matrix that the shape of column-major opA(B) is [k, n]
 * @param [in] ldb leading dimension of B
 * @param [in] beta host or device scalar
 * @param [inout] c_ptr column-major matrix of size [m, n]
 * @param [in] ldc leading dimension of C
 * @param [in] stream
 */
template <bool DevicePointerMode = false, typename S, typename A, typename B, typename C>
void matmul(raft::resources const& res,
            bool trans_a,
            bool trans_b,
            uint64_t m,
            uint64_t n,
            uint64_t k,
            const S* alpha,
            const A* a_ptr,
            uint64_t lda,
            const B* b_ptr,
            uint64_t ldb,
            const S* beta,
            C* c_ptr,
            uint64_t ldc,
            cudaStream_t stream)
{
  common::nvtx::range<common::nvtx::domain::raft> batch_scope(
    "linalg::matmul(m = %d, n = %d, k = %d)", m, n, k);
  std::shared_ptr<matmul_desc> mm_desc{nullptr};
  matmul_key_t mm_key{m, n, k, lda, ldb, ldc, trans_a, trans_b};
  static thread_local cache::
    lru<matmul_key_t, matmul_key_hash, std::equal_to<>, std::shared_ptr<matmul_desc>>
      cache{kLRUSize};
  if (!cache.get(mm_key, &mm_desc)) {
    mm_desc.reset(new matmul_desc{matmul_desc::create<S, A, B, C, DevicePointerMode>(res, mm_key)});
    cache.set(mm_key, mm_desc);
  }
  RAFT_CUBLAS_TRY(cublasLtMatmul(resource::get_cublaslt_handle(res),
                                 mm_desc->desc,
                                 alpha,
                                 a_ptr,
                                 mm_desc->a,
                                 b_ptr,
                                 mm_desc->b,
                                 beta,
                                 c_ptr,
                                 mm_desc->c,
                                 c_ptr,
                                 mm_desc->c,
                                 &(mm_desc->heuristics.algo),
                                 nullptr,
                                 0,
                                 stream));
}

template <typename T, bool DevicePointerMode = false>
void legacy_gemm(raft::resources const& res,
                 const bool trans_a,
                 const bool trans_b,
                 const int m,
                 const int n,
                 const int k,
                 const T* alpha,
                 const T* A,
                 const int lda,
                 const T* B,
                 const int ldb,
                 const T* beta,
                 T* C,
                 const int ldc,
                 cudaStream_t stream)
{
  return matmul<DevicePointerMode, T, T, T, T>(res,
                                               trans_a,
                                               trans_b,
                                               static_cast<uint64_t>(m),
                                               static_cast<uint64_t>(n),
                                               static_cast<uint64_t>(k),
                                               alpha,
                                               A,
                                               static_cast<uint64_t>(lda),
                                               B,
                                               static_cast<uint64_t>(ldb),
                                               beta,
                                               C,
                                               static_cast<uint64_t>(ldc),
                                               stream);
}

template <typename T>
void legacy_gemm(raft::resources const& res,
                 const T* a,
                 int n_rows_a,
                 int n_cols_a,
                 const T* b,
                 T* c,
                 int n_rows_c,
                 int n_cols_c,
                 cublasOperation_t trans_a,
                 cublasOperation_t trans_b,
                 T alpha,
                 T beta,
                 cudaStream_t stream)
{
  int m  = n_rows_c;
  int n  = n_cols_c;
  auto k = trans_a == CUBLAS_OP_T ? n_rows_a : n_cols_a;
  return matmul<false, T, T, T, T>(res,
                                   trans_a == CUBLAS_OP_T,
                                   trans_b == CUBLAS_OP_T,
                                   static_cast<uint64_t>(n_rows_c),
                                   static_cast<uint64_t>(n_cols_c),
                                   static_cast<uint64_t>(k),
                                   &alpha,
                                   a,
                                   static_cast<uint64_t>(trans_a == CUBLAS_OP_T ? k : m),
                                   b,
                                   static_cast<uint64_t>(trans_b == CUBLAS_OP_T ? n : k),
                                   &beta,
                                   c,
                                   static_cast<uint64_t>(m),
                                   stream);
}

template <typename T>
void legacy_gemm(raft::resources const& res,
                 const T* a,
                 int n_rows_a,
                 int n_cols_a,
                 const T* b,
                 T* c,
                 int n_rows_c,
                 int n_cols_c,
                 cublasOperation_t trans_a,
                 cublasOperation_t trans_b,
                 cudaStream_t stream)
{
  return legacy_gemm(
    res, a, n_rows_a, n_cols_a, b, c, n_rows_c, n_cols_c, trans_a, trans_b, T{1}, T{0}, stream);
}

template <typename T, bool DevicePointerMode = false>
void legacy_gemm(raft::resources const& res,
                 T* z,
                 T* x,
                 T* y,
                 int _M,
                 int _N,
                 int _K,
                 bool isZColMajor,
                 bool isXColMajor,
                 bool isYColMajor,
                 cudaStream_t stream,
                 const T* alpha,
                 const T* beta)
{
  if (isZColMajor) {
    return matmul<DevicePointerMode, T, T, T, T>(res,
                                                 !isXColMajor,
                                                 !isYColMajor,
                                                 static_cast<uint64_t>(_M),
                                                 static_cast<uint64_t>(_N),
                                                 static_cast<uint64_t>(_K),
                                                 alpha,
                                                 x,
                                                 static_cast<uint64_t>(isXColMajor ? _M : _K),
                                                 y,
                                                 static_cast<uint64_t>(isYColMajor ? _K : _N),
                                                 beta,
                                                 z,
                                                 static_cast<uint64_t>(_M),
                                                 stream);
  } else {
    return legacy_gemm<T, DevicePointerMode>(
      res, z, y, x, _N, _M, _K, true, !isYColMajor, !isXColMajor, stream, alpha, beta);
  }
}

}  // namespace raft::linalg::detail
