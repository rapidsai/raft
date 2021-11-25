/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <raft/cuda_utils.cuh>
#include <raft/pow2_utils.cuh>
#include <raft/vectorized.cuh>

namespace raft {
namespace linalg {

namespace linewise_impl {

template <typename Type, typename IdxType, std::size_t VecBytes, int BlockSize>
struct Linewise {
  static constexpr IdxType VecElems = VecBytes / sizeof(Type);

  typedef raft::TxN_t<Type, VecElems> Vec;
  typedef raft::Pow2<VecBytes> AlignBytes;
  typedef raft::Pow2<VecElems> AlignElems;
  typedef raft::Pow2<raft::WarpSize> AlignWarp;

  template <typename Lambda, typename... Vecs>
  static __device__ __forceinline__ void vectorCols(
    typename Vec::io_t* out, const typename Vec::io_t* in,
    const typename Vec::io_t* in_end, const IdxType rowLen, IdxType rowDiv,
    IdxType rowMod, Lambda op, Vecs... vecs) noexcept {
    constexpr IdxType warpPad = (AlignWarp::Value - 1) * VecElems;
    Type args[sizeof...(Vecs)];
    Vec v, w;
    bool update = true;
    for (; in < in_end;
         in += AlignWarp::Value, out += AlignWarp::Value, rowMod += warpPad) {
      v.val.internal = __ldcv(in);
      while (rowMod >= rowLen) {
        rowMod -= rowLen;
        rowDiv++;
        update = true;
      }
      if (update) {
        int l = 0;
        ((args[l] = vecs[rowDiv], l++), ...);
        update = false;
      }
#pragma unroll VecElems
      for (int k = 0; k < VecElems; k++, rowMod++) {
        if (rowMod == rowLen) {
          rowMod = 0;
          rowDiv++;
          int l = 0;
          ((args[l] = vecs[rowDiv], l++), ...);
        }
        int l = 0;
        w.val.data[k] = op(v.val.data[k], (std::ignore = vecs, args[l++])...);
      }
      *out = w.val.internal;
    }
  }

  template <typename Lambda, typename... Args>
  static __device__ __forceinline__ void vectorRows(
    typename Vec::io_t* out, const typename Vec::io_t* in, const IdxType len,
    Lambda op, Args... args) noexcept {
    Vec v;
    const IdxType d = BlockSize * gridDim.x;
    for (IdxType i = threadIdx.x + blockIdx.x * BlockSize; i < len; i += d) {
      v.val.internal = __ldcv(in + i);
#pragma unroll VecElems
      for (int k = 0; k < VecElems; k++)
        v.val.data[k] = op(v.val.data[k], args.val.data[k]...);
      __stwt(out + i, v.val.internal);
    }
  }

  static __device__ __forceinline__ Vec loadVec(const Type* p,
                                                const IdxType blockOffset,
                                                const IdxType rowLen) noexcept {
    __shared__ alignas(sizeof(Type) * VecElems) Type shm[VecElems * BlockSize];
    IdxType j = blockOffset + threadIdx.x;
#pragma unroll VecElems
    for (int k = threadIdx.x; k < VecElems * BlockSize;
         k += BlockSize, j += BlockSize) {
      while (j >= rowLen) j -= rowLen;
      shm[k] = p[j];
    }
    __syncthreads();
    {
      Vec out;
      out.val.internal =
        reinterpret_cast<typename Vec::io_t*>(shm)[threadIdx.x];
      return out;
    }
  }
};

template <typename Type, typename IdxType, std::size_t VecBytes, int BlockSize,
          typename Lambda, typename... Vecs>
__global__ void __launch_bounds__(BlockSize)
  matrixLinewiseVecColsMainKernel(Type* out, const Type* in,
                                  const IdxType arrOffset, const IdxType rowLen,
                                  const IdxType len,
                                  const IdxType elemsPerThread, Lambda op,
                                  Vecs... vecs) {
  typedef Linewise<Type, IdxType, VecBytes, BlockSize> L;

  IdxType t = L::AlignWarp::mod(threadIdx.x);
  t = arrOffset + elemsPerThread * (blockIdx.x * BlockSize + threadIdx.x - t) +
      t * L::VecElems;

  return L::vectorCols(
    reinterpret_cast<typename L::Vec::io_t*>(out + t),
    reinterpret_cast<const typename L::Vec::io_t*>(in + t),
    reinterpret_cast<const typename L::Vec::io_t*>(
      in + min(t + elemsPerThread * L::AlignWarp::Value, len)),
    rowLen, t / rowLen, t % rowLen, op, vecs...);
}

template <typename Type, typename IdxType, std::size_t MaxOffset,
          typename Lambda, typename... Vecs>
__global__ void __launch_bounds__(MaxOffset, 2)
  matrixLinewiseVecColsTailKernel(Type* out, const Type* in,
                                  const IdxType arrOffset,
                                  const IdxType arrTail, const IdxType rowLen,
                                  const IdxType len, Lambda op, Vecs... vecs) {
  typedef Linewise<Type, IdxType, sizeof(Type), MaxOffset> L;
  IdxType threadOffset, elemsPerWarp;
  if (blockIdx.x == 0) {
    threadOffset = threadIdx.x;
    elemsPerWarp = threadOffset < arrOffset;
  } else {
    threadOffset = arrTail + threadIdx.x;
    elemsPerWarp = threadOffset < len;
  }
  const IdxType rowDiv = threadOffset / rowLen;
  const IdxType rowMod = threadOffset % rowLen;
  return L::vectorCols(
    reinterpret_cast<typename L::Vec::io_t*>(out + threadOffset),
    reinterpret_cast<const typename L::Vec::io_t*>(in + threadOffset),
    reinterpret_cast<const typename L::Vec::io_t*>(in + threadOffset +
                                                   elemsPerWarp),
    rowLen, rowDiv, rowMod, op, vecs...);
}

template <typename Type, typename IdxType, std::size_t VecBytes, int BlockSize,
          typename Lambda, typename... Vecs>
__global__ void __launch_bounds__(BlockSize)
  matrixLinewiseVecRowsMainKernel(Type* out, const Type* in,
                                  const IdxType arrOffset, const IdxType rowLen,
                                  const IdxType len, Lambda op, Vecs... vecs) {
  typedef Linewise<Type, IdxType, VecBytes, BlockSize> L;
  const IdxType blockOffset =
    (arrOffset + BlockSize * L::VecElems * blockIdx.x) % rowLen;
  return L::vectorRows(reinterpret_cast<typename L::Vec::io_t*>(out),
                       reinterpret_cast<const typename L::Vec::io_t*>(in),
                       L::AlignElems::div(len), op,
                       L::loadVec(vecs, blockOffset, rowLen)...);
}

template <typename Type, typename IdxType, std::size_t MaxOffset,
          typename Lambda, typename... Vecs>
__global__ void __launch_bounds__(MaxOffset, 2)
  matrixLinewiseVecRowsTailKernel(Type* out, const Type* in,
                                  const IdxType arrOffset,
                                  const IdxType arrTail, const IdxType rowLen,
                                  const IdxType len, Lambda op, Vecs... vecs) {
  typedef Linewise<Type, IdxType, sizeof(Type), MaxOffset> L;
  if (blockIdx.x == 0)
    L::vectorRows(reinterpret_cast<typename L::Vec::io_t*>(out),
                  reinterpret_cast<const typename L::Vec::io_t*>(in), arrOffset,
                  op, L::loadVec(vecs, 0, rowLen)...);
  else
    L::vectorRows(
      reinterpret_cast<typename L::Vec::io_t*>(out + arrTail - MaxOffset),
      reinterpret_cast<const typename L::Vec::io_t*>(in + arrTail - MaxOffset),
      len - arrTail + MaxOffset, op,
      L::loadVec(vecs, arrTail % rowLen, rowLen)...);
}

template <typename Type, typename IdxType, std::size_t VecBytes,
          typename Lambda, typename... Vecs>
void matrixLinewiseVecCols(Type* out, const Type* in, const IdxType rowLen,
                           const IdxType nRows, Lambda op, cudaStream_t stream,
                           Vecs... vecs) {
  typedef raft::Pow2<VecBytes> AlignBytes;
  constexpr std::size_t VecElems = VecBytes / sizeof(Type);
  const IdxType totalLen = rowLen * nRows;
  const Type* alignedStart = AlignBytes::roundUp(in);
  const IdxType alignedOff = IdxType(alignedStart - in);
  const IdxType alignedEnd = IdxType(AlignBytes::roundDown(in + totalLen) - in);
  const IdxType alignedLen = alignedEnd - alignedOff;
  // blockSize
  constexpr int BlockSize = 256;
  constexpr dim3 bs(BlockSize, 1, 1);
  // Minimum size of the grid to make device well occupied
  const uint occupy = raft::getMultiProcessorCount() * 64;
  // does not make sense to have more blocks than this
  const uint maxBlocks = raft::ceildiv<uint>(uint(alignedLen), bs.x * VecElems);
  const dim3 gs(min(maxBlocks, occupy), 1, 1);

  const IdxType elemsPerThread =
    raft::ceildiv<IdxType>(alignedLen, gs.x * VecElems * BlockSize) * VecElems;
  matrixLinewiseVecColsMainKernel<Type, IdxType, VecBytes, BlockSize, Lambda,
                                  Vecs...><<<gs, bs, 0, stream>>>(
    out, in, alignedOff, rowLen, alignedLen, elemsPerThread, op, vecs...);
  CUDA_CHECK(cudaPeekAtLastError());
  if (alignedLen < totalLen) {
    // should be not smaller than the warp size for better branching
    constexpr std::size_t MaxOffset = std::max(std::size_t(32), VecBytes);
    matrixLinewiseVecColsTailKernel<Type, IdxType, MaxOffset, Lambda, Vecs...>
      <<<dim3(2, 1, 1), dim3(MaxOffset, 1, 1), 0, stream>>>(
        out, in, alignedOff, alignedEnd, rowLen, totalLen, op, vecs...);
    CUDA_CHECK(cudaPeekAtLastError());
  }
}

template <typename Type, typename IdxType, std::size_t VecBytes,
          typename Lambda, typename... Vecs>
void matrixLinewiseVecRows(Type* out, const Type* in, const IdxType rowLen,
                           const IdxType nRows, Lambda op, cudaStream_t stream,
                           Vecs... vecs) {
  typedef raft::Pow2<VecBytes> AlignBytes;
  constexpr std::size_t VecElems = VecBytes / sizeof(Type);
  const IdxType totalLen = rowLen * nRows;
  // blockSize
  constexpr int BlockSize = 256;
  constexpr dim3 bs(BlockSize, 1, 1);
  // if we have `stride` number of blocks, then each block processes always the same
  // indices along dimension rowLen; this means a block needs to index `vecs` only once!
  const uint stride =
    (rowLen / raft::gcd(bs.x * uint(VecElems), uint(rowLen))) * VecElems;
  // Minimum size of the grid to make device well occupied
  const uint occupy = raft::getMultiProcessorCount() * 64;
  const dim3 gs(min(
                  // does not make sense to have more blocks than this
                  raft::ceildiv<uint>(uint(totalLen), bs.x * VecElems),
                  // increase the stride size if necessary
                  raft::ceildiv<uint>(occupy, stride) * stride),
                1, 1);

  const Type* alignedStart = AlignBytes::roundUp(in);
  const IdxType alignedOff = IdxType(alignedStart - in);
  const IdxType alignedEnd = IdxType(AlignBytes::roundDown(in + totalLen) - in);
  const IdxType alignedLen = alignedEnd - alignedOff;
  matrixLinewiseVecRowsMainKernel<Type, IdxType, VecBytes, BlockSize, Lambda,
                                  Vecs...>
    <<<gs, bs, 0, stream>>>(out + alignedOff, alignedStart, alignedOff, rowLen,
                            alignedLen, op, vecs...);
  CUDA_CHECK(cudaPeekAtLastError());
  if (alignedLen < totalLen) {
    // should be not smaller than the warp size for better branching
    constexpr std::size_t MaxOffset = std::max(std::size_t(32), VecBytes);
    matrixLinewiseVecRowsTailKernel<Type, IdxType, MaxOffset, Lambda, Vecs...>
      <<<dim3(2, 1, 1), dim3(MaxOffset, 1, 1), 0, stream>>>(
        out, in, alignedOff, alignedEnd, rowLen, totalLen, op, vecs...);
    CUDA_CHECK(cudaPeekAtLastError());
  }
}

template <std::size_t VecBytes = 16>
struct MatrixLinewiseOp {
  template <typename Type, typename IdxType, typename Lambda, typename... Vecs>
  static void run(Type* out, const Type* in, const IdxType lineLen,
                  const IdxType nLines, const bool alongLines, Lambda op,
                  cudaStream_t stream, Vecs... vecs) {
    if constexpr (VecBytes > sizeof(Type)) {
      if (!raft::Pow2<VecBytes>::areSameAlignOffsets(in, out))
        return MatrixLinewiseOp<std::max((VecBytes >> 1), sizeof(Type))>::run(
          out, in, lineLen, nLines, alongLines, op, stream, vecs...);
    }
    if (alongLines)
      return matrixLinewiseVecRows<Type, IdxType, VecBytes, Lambda, Vecs...>(
        out, in, lineLen, nLines, op, stream, vecs...);
    else
      return matrixLinewiseVecCols<Type, IdxType, VecBytes, Lambda, Vecs...>(
        out, in, lineLen, nLines, op, stream, vecs...);
  }
};

};  // namespace linewise_impl

/**
 * Run a function over matrix lines (rows or columns) with a variable number
 * row-vectors or column-vectors.
 * The term `line` here signifies that the lines can be either columns or rows,
 * depending on the matrix layout.
 * What matters is if vectors are applied along lines (indices of vectors correspond
 * indices within lines), or across lines (indices of vectors correspond to line indices).
 *
 * @param out result of the operation; can be same as `in`; should be aligned the same as `in`
 *        to allow faster vectorized memory transfers.
 * @param in input matrix consisting of `nLines` lines, each `lineLen`-long.
 * @param lineLen length of matrix line in elements (`=nCols` in row-major or `=nRows` in col-major)
 * @param nLines number of matrix lines (`=nRows` in row-major or `=nCols` in col-major)
 * @param alongLines whether vectors are indices along or across lines.
 * @param op the operation applied on each line:
 *    for i in [0..lineLen) and j in [0..nLines):
 *      out[i, j] = op(in[i, j], vec1[i], vec2[i], ... veck[i])   if alongLines = true
 *      out[i, j] = op(in[i, j], vec1[j], vec2[j], ... veck[j])   if alongLines = false
 *    where matrix indexing is row-major ([i, j] = [i + lineLen * j]).
 * @param stream a cuda stream for the kernels
 * @param vecs zero or more vectors to be passed as arguments,
 *    size of each vector is `alongLines ? lineLen : nLines`.
 */
template <typename Type, typename IdxType, typename Lambda, typename... Vecs>
void matrixLinewiseOp(Type* out, const Type* in, const IdxType lineLen,
                      const IdxType nLines, const bool alongLines, Lambda op,
                      cudaStream_t stream, Vecs... vecs) {
  linewise_impl::MatrixLinewiseOp<16>::run<Type, IdxType, Lambda, Vecs...>(
    out, in, lineLen, nLines, alongLines, op, stream, vecs...);
}

};  // end namespace linalg
};  // end namespace raft
