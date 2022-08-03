/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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


/* This file is part of the Random Ball Cover (RBC) library.
 * (C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 */

#ifndef KERNELS_CU
#define KERNELS_CU

#include "defs.h"
#include <cuda.h>
#include <stdio.h>
namespace raft {
namespace spatial {
namespace knn {
namespace detail {

// min-max gate: it sets the minimum of x and y into x, the maximum into y, and
// exchanges the indices (xi and yi) accordingly.
__device__ __inline__ void mmGateI(real* x, real* y, unint* xi, unint* yi)
{
  int ti = MINi(*x, *y, *xi, *yi);
  *yi    = MAXi(*x, *y, *xi, *yi);
  *xi    = ti;
  real t = MIN(*x, *y);
  *y     = MAX(*x, *y);
  *x     = t;
}

//**************************************************************************
// The following functions are an implementation of Batcher's sorting network.
// All computations take place in (on-chip) shared memory.

// The function name is descriptive; it sorts each row of x, whose indices are xi.
__device__ __inline__ void sort16(real x[][16], unint xi[][16])
{
  int i = threadIdx.x;
  int j = threadIdx.y;

  if (i % 2 == 0) mmGateI(x[j] + i, x[j] + i + 1, xi[j] + i, xi[j] + i + 1);
  __syncthreads();

  if (i % 4 < 2) mmGateI(x[j] + i, x[j] + i + 2, xi[j] + i, xi[j] + i + 2);
  __syncthreads();

  if (i % 4 == 1) mmGateI(x[j] + i, x[j] + i + 1, xi[j] + i, xi[j] + i + 1);
  __syncthreads();

  if (i % 8 < 4) mmGateI(x[j] + i, x[j] + i + 4, xi[j] + i, xi[j] + i + 4);
  __syncthreads();

  if (i % 8 == 2 || i % 8 == 3) mmGateI(x[j] + i, x[j] + i + 2, xi[j] + i, xi[j] + i + 2);
  __syncthreads();

  if (i % 2 && i % 8 != 7) mmGateI(x[j] + i, x[j] + i + 1, xi[j] + i, xi[j] + i + 1);
  __syncthreads();

  // 0-7; 8-15 now sorted.  merge time.
  if (i < 8) mmGateI(x[j] + i, x[j] + i + 8, xi[j] + i, xi[j] + i + 8);
  __syncthreads();

  if (i > 3 && i < 8) mmGateI(x[j] + i, x[j] + i + 4, xi[j] + i, xi[j] + i + 4);
  __syncthreads();

  int os = (i / 2) * 4 + 2 + i % 2;
  if (i < 6) mmGateI(x[j] + os, x[j] + os + 2, xi[j] + os, xi[j] + os + 2);
  __syncthreads();

  if (i % 2 && i < 15) mmGateI(x[j] + i, x[j] + i + 1, xi[j] + i, xi[j] + i + 1);
}

// This function takes an array of lists, each of length 48. It is assumed
// that the first 32 numbers are sorted, and the last 16 numbers.  The
// routine then merges these lists into one sorted list of length 48.
__device__ __inline__ void merge32x16(real x[][48], unint xi[][48])
{
  int i = threadIdx.x;
  int j = threadIdx.y;

  mmGateI(x[j] + i, x[j] + i + 32, xi[j] + i, xi[j] + i + 32);
  __syncthreads();

  mmGateI(x[j] + i + 16, x[j] + i + 32, xi[j] + i + 16, xi[j] + i + 32);
  __syncthreads();

  int os = (i < 8) ? 24 : 0;
  mmGateI(x[j] + os + i, x[j] + os + i + 8, xi[j] + os + i, xi[j] + os + i + 8);
  __syncthreads();

  os = (i / 4) * 8 + 4 + i % 4;
  mmGateI(x[j] + os, x[j] + os + 4, xi[j] + os, xi[j] + os + 4);
  if (i < 4) mmGateI(x[j] + 36 + i, x[j] + 36 + i + 4, xi[j] + 36 + i, xi[j] + 36 + i + 4);
  __syncthreads();

  os = (i / 2) * 4 + 2 + i % 2;
  mmGateI(x[j] + os, x[j] + os + 2, xi[j] + os, xi[j] + os + 2);

  os = (i / 2) * 4 + 34 + i % 2;
  if (i < 6) mmGateI(x[j] + os, x[j] + os + 2, xi[j] + os, xi[j] + os + 2);
  __syncthreads();

  os = 2 * i + 1;
  mmGateI(x[j] + os, x[j] + os + 1, xi[j] + os, xi[j] + os + 1);

  os = 2 * i + 33;
  if (i < 7) mmGateI(x[j] + os, x[j] + os + 1, xi[j] + os, xi[j] + os + 1);
}

// This is the same as sort16, but takes as input lists of length 48
// and sorts the last 16 entries.  This cleans up some of the NN code,
// though it is inelegant.
__device__ __inline__ void sort16off(real x[][48], unint xi[][48])
{
  int i = threadIdx.x;
  int j = threadIdx.y;

  if (i % 2 == 0)
    mmGateI(x[j] + KMAX + i, x[j] + KMAX + i + 1, xi[j] + KMAX + i, xi[j] + KMAX + i + 1);
  __syncthreads();

  if (i % 4 < 2)
    mmGateI(x[j] + KMAX + i, x[j] + KMAX + i + 2, xi[j] + KMAX + i, xi[j] + KMAX + i + 2);
  __syncthreads();

  if (i % 4 == 1)
    mmGateI(x[j] + KMAX + i, x[j] + KMAX + i + 1, xi[j] + KMAX + i, xi[j] + KMAX + i + 1);
  __syncthreads();

  if (i % 8 < 4)
    mmGateI(x[j] + KMAX + i, x[j] + KMAX + i + 4, xi[j] + KMAX + i, xi[j] + KMAX + i + 4);
  __syncthreads();

  if (i % 8 == 2 || i % 8 == 3)
    mmGateI(x[j] + KMAX + i, x[j] + KMAX + i + 2, xi[j] + KMAX + i, xi[j] + KMAX + i + 2);
  __syncthreads();

  if (i % 2 && i % 8 != 7)
    mmGateI(x[j] + KMAX + i, x[j] + KMAX + i + 1, xi[j] + KMAX + i, xi[j] + KMAX + i + 1);
  __syncthreads();

  // 0-7; 8-15 now sorted.  merge time.
  if (i < 8) mmGateI(x[j] + KMAX + i, x[j] + KMAX + i + 8, xi[j] + KMAX + i, xi[j] + KMAX + i + 8);
  __syncthreads();

  if (i > 3 && i < 8)
    mmGateI(x[j] + KMAX + i, x[j] + KMAX + i + 4, xi[j] + KMAX + i, xi[j] + KMAX + i + 4);
  __syncthreads();

  int os = (i / 2) * 4 + 2 + i % 2;
  if (i < 6)
    mmGateI(x[j] + KMAX + os, x[j] + KMAX + os + 2, xi[j] + KMAX + os, xi[j] + KMAX + os + 2);
  __syncthreads();

  if (i % 2 && i < 15)
    mmGateI(x[j] + KMAX + i, x[j] + KMAX + i + 1, xi[j] + KMAX + i, xi[j] + KMAX + i + 1);
}

// This kernel does the same thing as nnKernel, except it only considers pairs as
// specified by the compPlan.
__global__ __inline__ void planNNKernel(const matrix Q,
                                        const unint* qMap,
                                        const matrix X,
                                        const intMatrix xMap,
                                        real* dMins,
                                        unint* dMinIDs,
                                        compPlan cP,
                                        unint qStartPos)
{
  unint qB = qStartPos + blockIdx.y * BLOCK_SIZE;  // indexes Q
  unint xB;                                        // X (DB) Block;
  unint cB;                                        // column Block
  unint offQ = threadIdx.y;                        // the offset of qPos in this block
  unint offX = threadIdx.x;                        // ditto for x
  unint i, j, k;
  unint groupIts;

  __shared__ real min[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ unint minPos[BLOCK_SIZE][BLOCK_SIZE];

  __shared__ real Xs[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ real Qs[BLOCK_SIZE][BLOCK_SIZE];

  unint g;   // query group of q
  unint xG;  // DB group currently being examined
  unint numGroups;
  unint groupCount;

  g         = cP.qToQGroup[qB];
  numGroups = cP.numGroups[g];

  min[offQ][offX] = MAX_REAL;
  __syncthreads();

  for (i = 0; i < numGroups; i++) {  // iterate over DB groups
    xG         = cP.qGroupToXGroup[IDX(g, i, cP.ld)];
    groupCount = cP.groupCountX[IDX(g, i, cP.ld)];
    groupIts   = (groupCount + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (j = 0; j < groupIts; j++) {  // iterate over elements of group
      xB = j * BLOCK_SIZE;

      real ans = 0;
      for (cB = 0; cB < X.pc; cB += BLOCK_SIZE) {  // iterate over cols to compute distances

        Xs[offX][offQ] = X.mat[IDX(xMap.mat[IDX(xG, xB + offQ, xMap.ld)], cB + offX, X.ld)];
        Qs[offX][offQ] =
          ((qMap[qB + offQ] == DUMMY_IDX) ? 0 : Q.mat[IDX(qMap[qB + offQ], cB + offX, Q.ld)]);
        __syncthreads();

        for (k = 0; k < BLOCK_SIZE; k++)
          ans += DIST(Xs[k][offX], Qs[k][offQ]);

        __syncthreads();
      }

      // compare to previous min and store into shared mem if needed.
      if (xB + offX < groupCount && ans < min[offQ][offX]) {
        min[offQ][offX]    = ans;
        minPos[offQ][offX] = xMap.mat[IDX(xG, xB + offX, xMap.ld)];
      }
      __syncthreads();
    }
  }

  // Reduce across threads
  for (i = BLOCK_SIZE / 2; i > 0; i /= 2) {
    if (offX < i) {
      if (min[offQ][offX + i] < min[offQ][offX]) {
        min[offQ][offX]    = min[offQ][offX + i];
        minPos[offQ][offX] = minPos[offQ][offX + i];
      }
    }
    __syncthreads();
  }

  if (offX == 0 && qMap[qB + offQ] != DUMMY_IDX) {
    dMins[qMap[qB + offQ]]   = min[offQ][0];
    dMinIDs[qMap[qB + offQ]] = minPos[offQ][0];
  }
}

// This is indentical to the planNNkernel, except that it maintains a list of 32-NNs.  At
// each iteration-chunk, the next 16 distances are computed, then sorted, then merged
// with the previously computed 32-NNs.
__global__ __inline__ void planKNNKernel(const matrix Q,
                                         const unint* qMap,
                                         const matrix X,
                                         const intMatrix xMap,
                                         matrix dMins,
                                         intMatrix dMinIDs,
                                         compPlan cP,
                                         unint qStartPos)
{
  unint qB = qStartPos + blockIdx.y * BLOCK_SIZE;  // indexes Q
  unint xB;                                        // X (DB) Block;
  unint cB;                                        // column Block
  unint offQ = threadIdx.y;                        // the offset of qPos in this block
  unint offX = threadIdx.x;                        // ditto for x
  unint i, j, k;
  unint groupIts;

  __shared__ real dNN[BLOCK_SIZE][KMAX + BLOCK_SIZE];
  __shared__ unint idNN[BLOCK_SIZE][KMAX + BLOCK_SIZE];

  __shared__ real Xs[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ real Qs[BLOCK_SIZE][BLOCK_SIZE];

  unint g;   // query group of q
  unint xG;  // DB group currently being examined
  unint numGroups;
  unint groupCount;

  g         = cP.qToQGroup[qB];
  numGroups = cP.numGroups[g];

  dNN[offQ][offX]       = MAX_REAL;
  dNN[offQ][offX + 16]  = MAX_REAL;
  idNN[offQ][offX]      = DUMMY_IDX;
  idNN[offQ][offX + 16] = DUMMY_IDX;
  __syncthreads();

  for (i = 0; i < numGroups; i++) {  // iterate over DB groups
    xG         = cP.qGroupToXGroup[IDX(g, i, cP.ld)];
    groupCount = cP.groupCountX[IDX(g, i, cP.ld)];
    groupIts   = (groupCount + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (j = 0; j < groupIts; j++) {  // iterate over elements of group
      xB = j * BLOCK_SIZE;

      real ans = 0;
      for (cB = 0; cB < X.pc; cB += BLOCK_SIZE) {  // iterate over cols to compute distances

        Xs[offX][offQ] = X.mat[IDX(xMap.mat[IDX(xG, xB + offQ, xMap.ld)], cB + offX, X.ld)];
        Qs[offX][offQ] =
          ((qMap[qB + offQ] == DUMMY_IDX) ? 0 : Q.mat[IDX(qMap[qB + offQ], cB + offX, Q.ld)]);
        __syncthreads();

        for (k = 0; k < BLOCK_SIZE; k++)
          ans += DIST(Xs[k][offX], Qs[k][offQ]);

        __syncthreads();
      }

      dNN[offQ][offX + 32] = (xB + offX < groupCount) ? ans : MAX_REAL;
      idNN[offQ][offX + 32] =
        (xB + offX < groupCount) ? xMap.mat[IDX(xG, xB + offX, xMap.ld)] : DUMMY_IDX;
      __syncthreads();

      sort16off(dNN, idNN);
      __syncthreads();

      merge32x16(dNN, idNN);
    }
  }
  __syncthreads();

  if (qMap[qB + offQ] != DUMMY_IDX) {
    dMins.mat[IDX(qMap[qB + offQ], offX, dMins.ld)]          = dNN[offQ][offX];
    dMins.mat[IDX(qMap[qB + offQ], offX + 16, dMins.ld)]     = dNN[offQ][offX + 16];
    dMinIDs.mat[IDX(qMap[qB + offQ], offX, dMins.ld)]        = idNN[offQ][offX];
    dMinIDs.mat[IDX(qMap[qB + offQ], offX + 16, dMinIDs.ld)] = idNN[offQ][offX + 16];
  }
}

// The basic 1-NN search kernel.
__global__ __inline__ void nnKernel(
  const matrix Q, unint numDone, const matrix X, real* dMins, unint* dMinIDs)
{
  unint qB = blockIdx.y * BLOCK_SIZE + numDone;  // indexes Q
  unint xB;                                      // indexes X;
  unint cB;                                      // colBlock
  unint offQ = threadIdx.y;                      // the offset of qPos in this block
  unint offX = threadIdx.x;                      // ditto for x
  unint i;
  real ans;

  __shared__ real min[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ unint minPos[BLOCK_SIZE][BLOCK_SIZE];

  __shared__ real Xs[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ real Qs[BLOCK_SIZE][BLOCK_SIZE];

  min[offQ][offX] = MAX_REAL;
  __syncthreads();

  for (xB = 0; xB < X.pr; xB += BLOCK_SIZE) {
    ans = 0;
    for (cB = 0; cB < X.pc; cB += BLOCK_SIZE) {
      // Each thread loads one element of X and Q into memory.
      Xs[offX][offQ] = X.mat[IDX(xB + offQ, cB + offX, X.ld)];
      Qs[offX][offQ] = Q.mat[IDX(qB + offQ, cB + offX, Q.ld)];

      __syncthreads();

      for (i = 0; i < BLOCK_SIZE; i++)
        ans += DIST(Xs[i][offX], Qs[i][offQ]);

      __syncthreads();
    }

    if (xB + offX < X.r && ans < min[offQ][offX]) {
      minPos[offQ][offX] = xB + offX;
      min[offQ][offX]    = ans;
    }
  }
  __syncthreads();

  // reduce across threads
  for (i = BLOCK_SIZE / 2; i > 0; i /= 2) {
    if (offX < i) {
      if (min[offQ][offX + i] < min[offQ][offX]) {
        min[offQ][offX]    = min[offQ][offX + i];
        minPos[offQ][offX] = minPos[offQ][offX + i];
      }
    }
    __syncthreads();
  }

  if (offX == 0) {
    dMins[qB + offQ]   = min[offQ][0];
    dMinIDs[qB + offQ] = minPos[offQ][0];
  }
}

// Computes the 32-NNs for each query in Q.  It is similar to nnKernel above, but maintains a
// list of the 32 currently-closest points in the DB, instead of just the single NN.  After each
// batch of 16 points is processed, it sorts these 16 points according to the distance from the
// query, then merges this list with the other list.
__global__ __inline__ void knnKernel(
  const matrix Q, unint numDone, const matrix X, matrix dMins, intMatrix dMinIDs)
{
  unint qB = blockIdx.y * BLOCK_SIZE + numDone;  // indexes Q
  unint xB;                                      // indexes X;
  unint cB;                                      // colBlock
  unint offQ = threadIdx.y;                      // the offset of qPos in this block
  unint offX = threadIdx.x;                      // ditto for x
  unint i;
  real ans;

  __shared__ real Xs[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ real Qs[BLOCK_SIZE][BLOCK_SIZE];

  __shared__ real dNN[BLOCK_SIZE][KMAX + BLOCK_SIZE];
  __shared__ unint idNN[BLOCK_SIZE][KMAX + BLOCK_SIZE];

  dNN[offQ][offX]       = MAX_REAL;
  dNN[offQ][offX + 16]  = MAX_REAL;
  idNN[offQ][offX]      = DUMMY_IDX;
  idNN[offQ][offX + 16] = DUMMY_IDX;

  __syncthreads();

  for (xB = 0; xB < X.pr; xB += BLOCK_SIZE) {
    ans = 0;
    for (cB = 0; cB < X.pc; cB += BLOCK_SIZE) {
      // Each thread loads one element of X and Q into memory.
      Xs[offX][offQ] = X.mat[IDX(xB + offQ, cB + offX, X.ld)];
      Qs[offX][offQ] = Q.mat[IDX(qB + offQ, cB + offX, Q.ld)];
      __syncthreads();

      for (i = 0; i < BLOCK_SIZE; i++)
        ans += DIST(Xs[i][offX], Qs[i][offQ]);

      __syncthreads();
    }

    dNN[offQ][offX + 32]  = (xB + offX < X.r) ? ans : MAX_REAL;
    idNN[offQ][offX + 32] = xB + offX;
    __syncthreads();

    sort16off(dNN, idNN);
    __syncthreads();

    merge32x16(dNN, idNN);
  }
  __syncthreads();

  dMins.mat[IDX(qB + offQ, offX, dMins.ld)]        = dNN[offQ][offX];
  dMins.mat[IDX(qB + offQ, offX + 16, dMins.ld)]   = dNN[offQ][offX + 16];
  dMinIDs.mat[IDX(qB + offQ, offX, dMins.ld)]      = idNN[offQ][offX];
  dMinIDs.mat[IDX(qB + offQ, offX + 16, dMins.ld)] = idNN[offQ][offX + 16];
}

__global__ __inline__ void sumKernel(charMatrix in, intMatrix sum, intMatrix sumaux, unint n)
{
  unint id = threadIdx.x;
  unint bo = blockIdx.x * SCAN_WIDTH;  // block offset
  unint r  = blockIdx.y;
  unint d, t;

  const unint l = SCAN_WIDTH;  // length

  unint off = 1;

  __shared__ unint ssum[l];

  ssum[2 * id]     = (bo + 2 * id < n) ? in.mat[IDX(r, bo + 2 * id, in.ld)] : 0;
  ssum[2 * id + 1] = (bo + 2 * id + 1 < n) ? in.mat[IDX(r, bo + 2 * id + 1, in.ld)] : 0;

  // up-sweep
  for (d = l >> 1; d > 0; d >>= 1) {
    __syncthreads();

    if (id < d) { ssum[off * (2 * id + 2) - 1] += ssum[off * (2 * id + 1) - 1]; }
    off *= 2;
  }

  __syncthreads();

  if (id == 0) {
    sumaux.mat[IDX(r, blockIdx.x, sumaux.ld)] = ssum[l - 1];
    ssum[l - 1]                               = 0;
  }

  // down-sweep
  for (d = 1; d < l; d *= 2) {
    off >>= 1;
    __syncthreads();

    if (id < d) {
      t                            = ssum[off * (2 * id + 1) - 1];
      ssum[off * (2 * id + 1) - 1] = ssum[off * (2 * id + 2) - 1];
      ssum[off * (2 * id + 2) - 1] += t;
    }
  }

  __syncthreads();

  if (bo + 2 * id < n) sum.mat[IDX(r, bo + 2 * id, sum.ld)] = ssum[2 * id];
  if (bo + 2 * id + 1 < n) sum.mat[IDX(r, bo + 2 * id + 1, sum.ld)] = ssum[2 * id + 1];
}

// This is the same as sumKernel, but takes an int matrix as input.
__global__ __inline__ void sumKernelI(intMatrix in, intMatrix sum, intMatrix sumaux, unint n)
{
  unint id = threadIdx.x;
  unint bo = blockIdx.x * SCAN_WIDTH;  // block offset
  unint r  = blockIdx.y;
  unint d, t;

  const unint l = SCAN_WIDTH;  // length

  unint off = 1;

  __shared__ unint ssum[l];

  ssum[2 * id]     = (bo + 2 * id < n) ? in.mat[IDX(r, bo + 2 * id, in.ld)] : 0;
  ssum[2 * id + 1] = (bo + 2 * id + 1 < n) ? in.mat[IDX(r, bo + 2 * id + 1, in.ld)] : 0;

  // up-sweep
  for (d = l >> 1; d > 0; d >>= 1) {
    __syncthreads();

    if (id < d) { ssum[off * (2 * id + 2) - 1] += ssum[off * (2 * id + 1) - 1]; }
    off *= 2;
  }

  __syncthreads();

  if (id == 0) {
    sumaux.mat[IDX(r, blockIdx.x, sumaux.ld)] = ssum[l - 1];
    ssum[l - 1]                               = 0;
  }

  // down-sweep
  for (d = 1; d < l; d *= 2) {
    off >>= 1;
    __syncthreads();

    if (id < d) {
      t                            = ssum[off * (2 * id + 1) - 1];
      ssum[off * (2 * id + 1) - 1] = ssum[off * (2 * id + 2) - 1];
      ssum[off * (2 * id + 2) - 1] += t;
    }
  }

  __syncthreads();

  if (bo + 2 * id < n) sum.mat[IDX(r, bo + 2 * id, sum.ld)] = ssum[2 * id];

  if (bo + 2 * id + 1 < n) sum.mat[IDX(r, bo + 2 * id + 1, sum.ld)] = ssum[2 * id + 1];
}

// Computes all pairs of distances between Q and X.
__global__ __inline__ void dist1Kernel(
  const matrix Q, unint qStart, const matrix X, unint xStart, matrix D)
{
  unint c, i, j;

  unint qB = blockIdx.y * BLOCK_SIZE + qStart;
  unint q  = threadIdx.y;
  unint xB = blockIdx.x * BLOCK_SIZE + xStart;
  unint x  = threadIdx.x;

  real ans = 0;

  // This thread is responsible for computing the dist between Q[qB+q] and X[xB+x]

  __shared__ real Qs[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ real Xs[BLOCK_SIZE][BLOCK_SIZE];

  for (i = 0; i < Q.pc / BLOCK_SIZE; i++) {
    c = i * BLOCK_SIZE;  // current col block

    Qs[x][q] = Q.mat[IDX(qB + q, c + x, Q.ld)];
    Xs[x][q] = X.mat[IDX(xB + q, c + x, X.ld)];

    __syncthreads();

    for (j = 0; j < BLOCK_SIZE; j++)
      ans += DIST(Qs[j][q], Xs[j][x]);

    __syncthreads();
  }

  D.mat[IDX(qB + q, xB + x, D.ld)] = ans;
}

// This function is used by the rbc building routine.  It find an appropriate range
// such that roughly cntWant points fall within this range.  D is a matrix of distances.
__global__ __inline__ void findRangeKernel(const matrix D,
                                           unint numDone,
                                           real* ranges,
                                           unint cntWant)
{
  unint row = blockIdx.y * (BLOCK_SIZE / 4) + threadIdx.y + numDone;
  unint ro  = threadIdx.y;
  unint co  = threadIdx.x;
  unint i, c;
  real t;

  const unint LB = (90 * cntWant) / 100;
  const unint UB = cntWant;

  __shared__ real smin[BLOCK_SIZE / 4][4 * BLOCK_SIZE];
  __shared__ real smax[BLOCK_SIZE / 4][4 * BLOCK_SIZE];

  real min = MAX_REAL;
  real max = 0;
  for (c = 0; c < D.pc; c += (4 * BLOCK_SIZE)) {
    if (c + co < D.c) {
      t   = D.mat[IDX(row, c + co, D.ld)];
      min = MIN(t, min);
      max = MAX(t, max);
    }
  }

  smin[ro][co] = min;
  smax[ro][co] = max;
  __syncthreads();

  for (i = 2 * BLOCK_SIZE; i > 0; i /= 2) {
    if (co < i) {
      smin[ro][co] = MIN(smin[ro][co], smin[ro][co + i]);
      smax[ro][co] = MAX(smax[ro][co], smax[ro][co + i]);
    }
    __syncthreads();
  }

  // Now start range counting.

  unint itcount = 0;
  unint cnt;
  real rg;
  __shared__ unint scnt[BLOCK_SIZE / 4][4 * BLOCK_SIZE];
  __shared__ char cont[BLOCK_SIZE / 4];

  if (co == 0) cont[ro] = 1;

  do {
    itcount++;
    __syncthreads();

    if (cont[ro])  // if we didn't actually need to cont, leave rg as it was.
      rg = (smax[ro][0] + smin[ro][0]) / ((real)2.0);

    cnt = 0;
    for (c = 0; c < D.pc; c += (4 * BLOCK_SIZE)) {
      cnt += (c + co < D.c && row < D.r && D.mat[IDX(row, c + co, D.ld)] <= rg);
    }

    scnt[ro][co] = cnt;
    __syncthreads();

    for (i = 2 * BLOCK_SIZE; i > 0; i /= 2) {
      if (co < i) { scnt[ro][co] += scnt[ro][co + i]; }
      __syncthreads();
    }

    if (co == 0) {
      if (scnt[ro][0] < cntWant)
        smin[ro][0] = rg;
      else
        smax[ro][0] = rg;
    }

    // cont[ro] == this row needs to continue
    if (co == 0) cont[ro] = row < D.r && (scnt[ro][0] < LB || scnt[ro][0] > UB);
    __syncthreads();

    // Determine if *any* of the rows need to continue
    for (i = BLOCK_SIZE / 8; i > 0; i /= 2) {
      if (ro < i && co == 0) cont[ro] |= cont[ro + i];
      __syncthreads();
    }

  } while (cont[0]);

  if (co == 0 && row < D.r) ranges[row] = rg;
}

__global__ __inline__ void rangeSearchKernel(
  const matrix D, unint xOff, unint yOff, const real* ranges, charMatrix ir)
{
  unint col = blockIdx.x * BLOCK_SIZE + threadIdx.x + xOff;
  unint row = blockIdx.y * BLOCK_SIZE + threadIdx.y + yOff;

  ir.mat[IDX(row, col, ir.ld)] = D.mat[IDX(row, col, D.ld)] < ranges[row];
}

__global__ __inline__ void rangeCountKernel(
  const matrix Q, unint numDone, const matrix X, real* ranges, unint* counts)
{
  unint q  = blockIdx.y * BLOCK_SIZE + numDone;
  unint qo = threadIdx.y;
  unint xo = threadIdx.x;

  real rg = ranges[q + qo];

  unint r, c, i;

  __shared__ unint scnt[BLOCK_SIZE][BLOCK_SIZE];

  __shared__ real xs[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ real qs[BLOCK_SIZE][BLOCK_SIZE];

  unint cnt = 0;
  for (r = 0; r < X.pr; r += BLOCK_SIZE) {
    real dist = 0;
    for (c = 0; c < X.pc; c += BLOCK_SIZE) {
      xs[xo][qo] = X.mat[IDX(r + qo, c + xo, X.ld)];
      qs[xo][qo] = Q.mat[IDX(q + qo, c + xo, Q.ld)];
      __syncthreads();

      for (i = 0; i < BLOCK_SIZE; i++)
        dist += DIST(xs[i][xo], qs[i][qo]);

      __syncthreads();
    }
    cnt += r + xo < X.r && dist < rg;
  }

  scnt[qo][xo] = cnt;
  __syncthreads();

  for (i = BLOCK_SIZE / 2; i > 0; i /= 2) {
    if (xo < i) { scnt[qo][xo] += scnt[qo][xo + i]; }
    __syncthreads();
  }

  if (xo == 0 && q + qo < Q.r) counts[q + qo] = scnt[qo][0];
}

__global__ __inline__ void combineSumKernel(intMatrix sum, unint numDone, intMatrix daux, unint n)
{
  unint id = threadIdx.x;
  unint bo = blockIdx.x * SCAN_WIDTH;
  unint r  = blockIdx.y + numDone;

  if (bo + 2 * id < n)
    sum.mat[IDX(r, bo + 2 * id, sum.ld)] += daux.mat[IDX(r, blockIdx.x, daux.ld)];
  if (bo + 2 * id + 1 < n)
    sum.mat[IDX(r, bo + 2 * id + 1, sum.ld)] += daux.mat[IDX(r, blockIdx.x, daux.ld)];
}

__global__ __inline__ void getCountsKernel(unint* counts,
                                           unint numDone,
                                           charMatrix ir,
                                           intMatrix sums)
{
  unint r = blockIdx.x * BLOCK_SIZE + threadIdx.x + numDone;
  if (r < ir.r) {
    counts[r] = ir.mat[IDX(r, ir.c - 1, ir.ld)] ? sums.mat[IDX(r, sums.c - 1, sums.ld)] + 1
                                                : sums.mat[IDX(r, sums.c - 1, sums.ld)];
  }
}

__global__ __inline__ void buildMapKernel(intMatrix map,
                                          charMatrix ir,
                                          intMatrix sums,
                                          unint offSet)
{
  unint id = threadIdx.x;
  unint bo = blockIdx.x * SCAN_WIDTH;
  unint r  = blockIdx.y;

  if (bo + 2 * id < ir.c && ir.mat[IDX(r, bo + 2 * id, ir.ld)])
    map.mat[IDX(r + offSet, sums.mat[IDX(r, bo + 2 * id, sums.ld)], map.ld)] = bo + 2 * id;
  if (bo + 2 * id + 1 < ir.c && ir.mat[IDX(r, bo + 2 * id + 1, ir.ld)])
    map.mat[IDX(r + offSet, sums.mat[IDX(r, bo + 2 * id + 1, sums.ld)], map.ld)] = bo + 2 * id + 1;
}

}  // namespace detail
}  // namespace knn
}  // namespace spatial
};  // namespace raft
#endif
