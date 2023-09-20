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
#ifndef __CONTRACTIONS_H
#define __CONTRACTIONS_H

#pragma once

#include "detail/contractions.cuh"

namespace raft {
namespace linalg {

/**
 * @brief This is the central enum that should be used to configure the perf
 *        landscape of the Contraction kernel.
 *
 * Main goal of this Policy struct is to provide sufficient knobs to tune the
 * perf of Contraction kernel, as and when we see matrices of different shapes.
 *
 * @tparam DataT   the IO and math datatype
 * @tparam _veclen number of k-elements loaded by each thread for every LDG call
 *                 it makes. This should be configured based on the input 'k'
 *                 value and the input data type. For eg: if DataT = float and
 *                 k is multiples of 4, then setting this to 4 gives the best
 *                 LDG pattern. Possible values are {1, 2, 4}.
 * @tparam _kblk   number of k-elements operated upon per main-loop iteration.
 *                 Therefore total number of main-loop iterations will be
 *                 `ceil(k/_kblk)`. This must be multiples of `_veclen`. Do note
 *                 that bigger this value, the greater shared mem requirement.
 * @tparam _rpt    Defines the number of rows that a given thread accumulates on.
 *                 This directly results in increased register pressure. This
 *                 also is used to compute the number of m-elements worked upon
 *                 by each thread block.
 * @tparam _cpt    Defines the number of cols that a given thread accumulates on.
 *                 This directly results in increased register pressure. This
 *                 also is used to compute the number of n-elements worked upon
 *                 by each thread block.
 * @tparam _tr     Number of threads working on the same output column. This is
 *                 used to compute the number of m-elements worked upon by each
 *                 thread block. This also determines the number of threads per
 *                 thread block
 * @tparam _tc     Number of threads working on the same output row. This is
 *                 used to compute the number of m-elements worked upon by each
 *                 thread block. This also determines the number of threads per
 *                 thread block
 */
template <typename DataT, int _veclen, int _kblk, int _rpt, int _cpt, int _tr, int _tc>
struct KernelPolicy {
  enum {
    /** number of elements along K worked upon per main loop iteration */
    Kblk = _kblk,
    /** number of elements loaded per LDG */
    Veclen = _veclen,
    /** number of rows a thread works on for accumulation */
    AccRowsPerTh = _rpt,
    /** number of cols a thread works on for accumulation */
    AccColsPerTh = _cpt,
    /** number of threads working the same output col */
    AccThRows = _tr,
    /** number of threads working the same output row */
    AccThCols = _tc,
    /** total threads per block */
    Nthreads = AccThRows * AccThCols,
    /** output tile size along rows */
    Mblk = AccRowsPerTh * AccThRows,
    /** output tile size along cols */
    Nblk = AccColsPerTh * AccThCols,
    /** number of threads loading a single row */
    LdgThRow = Kblk / Veclen,
    /** number of LDGs issued by a single thread for X */
    LdgPerThX = Mblk * LdgThRow / Nthreads,
    /** number of LDGs issued by a single thread for Y */
    LdgPerThY = Nblk * LdgThRow / Nthreads,
    /** number of rows of X covered per LDG */
    LdgRowsX = Mblk / LdgPerThX,
    /** number of rows of Y covered per LDG */
    LdgRowsY = Nblk / LdgPerThY,
    /** stride for accessing X/Y data in shared mem */
    SmemStride = Kblk + Veclen,
    /** size of one page for storing X data */
    SmemPageX = SmemStride * Mblk,
    /** size of one page for storing Y data */
    SmemPageY = SmemStride * Nblk,
    /** size of one smem page */
    SmemPage = SmemPageX + SmemPageY,
    /** size (in B) for smem needed */
    SmemSize = 2 * SmemPage * sizeof(DataT),
  };  // enum

};  // struct KernelPolicy

template <typename DataT, int _veclen, int _kblk, int _rpt, int _cpt, int _tr, int _tc>
struct ColKernelPolicy {
  enum {
    /** number of elements along K worked upon per main loop iteration */
    Kblk = _kblk,
    /** number of elements loaded per LDG */
    Veclen = _veclen,
    /** number of rows a thread works on for accumulation */
    AccRowsPerTh = _rpt,
    /** number of cols a thread works on for accumulation */
    AccColsPerTh = _cpt,
    /** number of threads working the same output col */
    AccThRows = _tr,
    /** number of threads working the same output row */
    AccThCols = _tc,
    /** total threads per block */
    Nthreads = AccThRows * AccThCols,
    /** output tile size along rows */
    Mblk = AccRowsPerTh * AccThRows,
    /** output tile size along cols */
    Nblk = AccColsPerTh * AccThCols,
    /** number of threads loading a single col */
    LdgThRow = Mblk / Veclen,
    /** number of LDGs issued by a single thread for X */
    LdgPerThX = Kblk * LdgThRow / Nthreads,
    /** number of LDGs issued by a single thread for Y */
    LdgPerThY = Kblk * LdgThRow / Nthreads,
    /** number of rows of X covered per LDG */
    LdgRowsX = Kblk / LdgPerThX,
    /** number of rows of Y covered per LDG */
    LdgRowsY = Kblk / LdgPerThY,
    /** stride for accessing X/Y data in shared mem */
    SmemStride = Mblk + Veclen,
    /** size of one page for storing X data */
    SmemPageX = SmemStride * Kblk,
    /** size of one page for storing Y data */
    SmemPageY = SmemStride * Kblk,
    /** size of one smem page */
    SmemPage = SmemPageX + SmemPageY,
    /** size (in B) for smem needed */
    SmemSize = 2 * SmemPage * sizeof(DataT),
  };  // colMajor enum
  static_assert(Mblk == Nblk, "Mblk should be equal to Nblk");
};
/**
 * @defgroup Policy4x4 16 elements per thread Policy with k-block = 32
 * @{
 */
template <typename DataT, int _veclen>
struct Policy4x4 {};

template <int _veclen>
struct Policy4x4<float, _veclen> {
  typedef KernelPolicy<float, _veclen, 32, 4, 4, 16, 16> Policy;
  typedef ColKernelPolicy<float, _veclen, 32, 4, 4, 16, 16> ColPolicy;
};

template <int _veclen>
struct Policy4x4<double, _veclen> {
  typedef KernelPolicy<double, _veclen, 16, 4, 4, 16, 16> Policy;
  typedef ColKernelPolicy<double, _veclen, 16, 4, 4, 16, 16> ColPolicy;
};
/** @} */

/**
 * A smaller k-block (8 instead of 32) with fewer threads per block (8x8 instead
 * of 16x16), which is faster for raft::distance::fusedL2NN on skinny matrices,
 * i.e., matrices with a small k dimension.
 *
 */
template <typename DataT, int _veclen>
struct Policy4x4Skinny {};

template <int _veclen>
struct Policy4x4Skinny<float, _veclen> {
  typedef KernelPolicy<float, _veclen, 8, 4, 4, 8, 8> Policy;
  typedef ColKernelPolicy<float, _veclen, 8, 4, 4, 8, 8> ColPolicy;
};

template <int _veclen>
struct Policy4x4Skinny<double, _veclen> {
  typedef KernelPolicy<double, _veclen, 8, 4, 4, 8, 8> Policy;
  typedef ColKernelPolicy<double, _veclen, 8, 4, 4, 8, 8> ColPolicy;
};

/**
 * @defgroup Policy2x8 16 elements per thread Policy with k-block = 16
 * @{
 */
template <typename DataT, int _veclen = 1>
struct Policy2x8 {};

template <int _veclen>
struct Policy2x8<float, _veclen> {
  typedef KernelPolicy<float, _veclen, 16, 2, 8, 8, 32> Policy;
};

template <int _veclen>
struct Policy2x8<double, _veclen> {
  // this is not used just for keeping compiler happy.
  typedef KernelPolicy<double, _veclen, 32, 1, 2, 8, 32> Policy;
};
/** @} */

/**
 * @brief Base class for gemm-like NT contractions
 *
 * This class does not provide any arithmetic operations, but only provides the
 * memory-related operations of loading the `x` and `y` matrix blocks from the
 * global memory into shared memory and then from shared into registers. Thus,
 * this class acts as a basic building block for further composing gemm-like NT
 * contractions on input matrices which are row-major (and so does the output)
 *
 * @tparam DataT  IO and math data type
 * @tparam IdxT   indexing type
 * @tparam Policy policy used to customize memory access behavior.
 *                See documentation for `KernelPolicy` to know more.
 */
using detail::Contractions_NT;

}  // namespace linalg
}  // namespace raft

#endif
