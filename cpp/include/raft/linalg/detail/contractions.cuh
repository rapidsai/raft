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

#include <raft/util/device_loads_stores.cuh>

namespace raft {
namespace linalg {
namespace detail {

template <typename DataT, typename IdxT, typename Policy, bool isRowMajor = true>
struct Contractions_NT {
 protected:
  typedef Policy P;

  /** number of rows in X */
  IdxT m;
  /** number of rows in Y */
  IdxT n;
  /** number of columns in X and Y */
  IdxT k;
  /** leading dimension in X */
  IdxT lda;
  /** leading dimension in Y */
  IdxT ldb;
  /** leading dimension in Output D */
  IdxT ldd;

  /** global memory pointer to X matrix */
  const DataT* x_base;
  /** global memory pointer to Y matrix */
  const DataT* y_base;

  /** current thread's smem row id */
  int srowid;
  /** current thread's smem column id */
  int scolid;
  /** current thread's accumulation row id */
  int accrowid;
  /** current thread's accumulation column id */
  int acccolid;

  /** base smem pointer for X data storage */
  DataT* sx;
  /** base smem pointer for Y data storage */
  DataT* sy;
  /** index pointing the correct smem page for writing after `ldgXY()` */
  int pageWr;
  /** index pointing the correct smem page for reading during `ldsXY()` */
  int pageRd;

  /** block of X data loaded from smem after `ldsXY()` */
  DataT regx[P::AccRowsPerTh][P::Veclen];
  /** block of Y data loaded from smem after `ldsXY()` */
  DataT regy[P::AccColsPerTh][P::Veclen];
  /** block of X data loaded from global mem after `ldgXY()` */
  DataT ldgDataX[P::LdgPerThX][P::Veclen];
  /** block of Y data loaded from global mem after `ldgXY()` */
  DataT ldgDataY[P::LdgPerThY][P::Veclen];

  static constexpr DataT Zero = (DataT)0;

 public:
  /**
   * @brief Ctor
   * @param[in] _x X matrix. [on device] [dim = _m x _k] [row-major]
   * @param[in] _y Y matrix. [on device] [dim = _n x _k] [row-major]
   * @param[in] _m number of rows of X
   * @param[in] _n number of rows of Y
   * @param[in] _k number of cols of X and Y
   * @param[in] _smem shared memory region used during computations
   */
  DI Contractions_NT(const DataT* _x, const DataT* _y, IdxT _m, IdxT _n, IdxT _k, char* _smem)
    : m(_m),
      n(_n),
      k(_k),
      lda(_k),
      ldb(_k),
      x_base(_x),
      y_base(_y),
      srowid(threadIdx.x / P::LdgThRow),
      scolid((threadIdx.x % P::LdgThRow) * P::Veclen),
      accrowid(threadIdx.x / P::AccThCols),
      acccolid(threadIdx.x % P::AccThCols),
      sx((DataT*)_smem),
      sy(&(sx[P::SmemPageX])),
      pageWr(0),
      pageRd(0)
  {
  }

  /**
   * @brief Ctor
   * @param[in] _x X matrix. [on device] [dim = _m x _k] [row-major]
   * @param[in] _y Y matrix. [on device] [dim = _n x _k] [row-major]
   * @param[in] _m number of rows of X
   * @param[in] _n number of rows of Y
   * @param[in] _k number of cols of X and Y
   * @param[in] _smem shared memory region used during computations
   */
  DI Contractions_NT(const DataT* _x,
                     const DataT* _y,
                     IdxT _m,
                     IdxT _n,
                     IdxT _k,
                     IdxT _lda,
                     IdxT _ldb,
                     IdxT _ldd,
                     char* _smem)
    : m(_m),
      n(_n),
      k(_k),
      lda(_lda),
      ldb(_ldb),
      ldd(_ldd),
      x_base(_x),
      y_base(_y),
      srowid(threadIdx.x / P::LdgThRow),
      scolid((threadIdx.x % P::LdgThRow) * P::Veclen),
      accrowid(threadIdx.x / P::AccThCols),
      acccolid(threadIdx.x % P::AccThCols),
      sx((DataT*)_smem),
      sy(&(sx[P::SmemPageX])),
      pageWr(0),
      pageRd(0)
  {
  }

 protected:
  /**
   * @brief Load current block of X/Y from global memory to registers
   * @param[in] kidx current start index of k to be loaded
   */
  DI void ldgXY(IdxT tile_idx_m, IdxT tile_idx_n, IdxT kidx)
  {
    ldgX(tile_idx_m, kidx);
    ldgY(tile_idx_n, kidx);
  }

  DI void ldgXY(IdxT tile_idx_m, IdxT tile_idx_n, IdxT kidx, IdxT tile_end_n)
  {
    ldgX(tile_idx_m, kidx);
    ldgY(tile_idx_n, kidx, tile_end_n);
  }

  /**
   * @brief Store current block of X/Y from registers to smem
   * @param[in] kidx current start index of k to be loaded
   */
  DI void stsXY()
  {
    stsX(sx + pageWr * P::SmemPage);
    stsY(sy + pageWr * P::SmemPage);
  }

  /**
   * @brief Load X and Y block from shared memory to registers
   * @param[in] kidx k value from the current k-block to be loaded from smem
   */
  DI void ldsXY(int kidx)
  {
    ldsX(kidx, sx + pageRd * P::SmemPage);
    ldsY(kidx, sy + pageRd * P::SmemPage);
  }

  DI void switch_read_buffer() { this->pageRd ^= 1; }

  DI void switch_write_buffer() { this->pageWr ^= 1; }

 private:
  DI void ldgX(IdxT tile_idx_m, IdxT kidx)
  {
    IdxT xrowid = isRowMajor ? tile_idx_m + srowid : tile_idx_m;
    auto x      = isRowMajor ? x_base + xrowid * lda : x_base + xrowid + srowid * lda;

    if (isRowMajor) {
      auto numRows = m;
      auto koffset = kidx + scolid;
#pragma unroll
      for (int i = 0; i < P::LdgPerThX; ++i) {
        if (koffset < lda && (xrowid + i * P::LdgRowsX) < numRows) {
          ldg(ldgDataX[i], x + i * P::LdgRowsX * lda + koffset);
        } else {
#pragma unroll
          for (int j = 0; j < P::Veclen; ++j) {
            ldgDataX[i][j] = Zero;
          }
        }
      }
    } else {
      const auto numRows = k;
      auto koffset       = scolid;
#pragma unroll
      for (int i = 0; i < P::LdgPerThX; ++i) {
        if ((koffset + xrowid) < lda && (srowid + kidx + i * P::LdgRowsX) < numRows) {
          ldg(ldgDataX[i], x + (kidx + i * P::LdgRowsX) * lda + koffset);
        } else {
#pragma unroll
          for (int j = 0; j < P::Veclen; ++j) {
            ldgDataX[i][j] = Zero;
          }
        }
      }
    }
  }

  DI void ldgY(IdxT tile_idx_n, IdxT kidx) { ldgY(tile_idx_n, kidx, n); }

  DI void ldgY(IdxT tile_idx_n, IdxT kidx, IdxT end_n)
  {
    IdxT yrowid = isRowMajor ? tile_idx_n + srowid : tile_idx_n;
    auto y      = isRowMajor ? y_base + yrowid * ldb : y_base + yrowid + srowid * ldb;

    if (isRowMajor) {
      auto numRows = end_n;
      auto koffset = kidx + scolid;
#pragma unroll
      for (int i = 0; i < P::LdgPerThY; ++i) {
        if (koffset < ldb && (yrowid + i * P::LdgRowsY) < numRows) {
          ldg(ldgDataY[i], y + i * P::LdgRowsY * ldb + koffset);
        } else {
#pragma unroll
          for (int j = 0; j < P::Veclen; ++j) {
            ldgDataY[i][j] = Zero;
          }
        }
      }
    } else {
      auto numRows = k;
      auto koffset = scolid;
#pragma unroll
      for (int i = 0; i < P::LdgPerThY; ++i) {
        if ((koffset + yrowid) < end_n && (srowid + kidx + i * P::LdgRowsY) < numRows) {
          ldg(ldgDataY[i], y + (kidx + i * P::LdgRowsY) * ldb + koffset);
        } else {
#pragma unroll
          for (int j = 0; j < P::Veclen; ++j) {
            ldgDataY[i][j] = Zero;
          }
        }
      }
    }
  }

  DI void stsX(DataT* smem)
  {
    auto* saddr = smem + srowid * P::SmemStride + scolid;
#pragma unroll
    for (int i = 0; i < P::LdgPerThX; ++i) {
      sts(saddr + i * P::LdgRowsX * P::SmemStride, ldgDataX[i]);
    }
  }

  DI void stsY(DataT* smem)
  {
    auto* saddr = smem + srowid * P::SmemStride + scolid;
#pragma unroll
    for (int i = 0; i < P::LdgPerThY; ++i) {
      sts(saddr + i * P::LdgRowsY * P::SmemStride, ldgDataY[i]);
    }
  }

  DI void ldsX(int kidx, DataT* smem)
  {
    if (isRowMajor) {
      auto* saddr = smem + accrowid * P::SmemStride + kidx;
#pragma unroll
      for (int i = 0; i < P::AccRowsPerTh; ++i) {
        lds(regx[i], saddr + i * P::AccThRows * P::SmemStride);
      }
    } else {
      auto* saddr = smem + accrowid + kidx * P::SmemStride;
#pragma unroll
      for (int i = 0; i < P::AccRowsPerTh; ++i) {
#pragma unroll
        for (int v = 0; v < P::Veclen; ++v) {
          regx[i][v] = saddr[i * P::AccThRows + v * P::SmemStride];
        }
      }
    }
  }

  DI void ldsY(int kidx, DataT* smem)
  {
    if (isRowMajor) {
      auto* saddr = smem + acccolid * P::SmemStride + kidx;
#pragma unroll
      for (int i = 0; i < P::AccColsPerTh; ++i) {
        lds(regy[i], saddr + i * P::AccThCols * P::SmemStride);
      }
    } else {
      auto* saddr = smem + acccolid + kidx * P::SmemStride;
#pragma unroll
      for (int i = 0; i < P::AccColsPerTh; ++i) {
#pragma unroll
        for (int v = 0; v < P::Veclen; ++v) {
          regy[i][v] = saddr[i * P::AccThCols + v * P::SmemStride];
        }
      }
    }
  }

};  // struct Contractions_NT

}  // namespace detail
}  // namespace linalg
}  // namespace raft
