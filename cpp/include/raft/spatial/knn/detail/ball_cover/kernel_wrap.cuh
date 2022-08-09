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

#ifndef KERNELWRAP_CU
#define KERNELWRAP_CU

#include "defs.h"
#include "kernels.cuh"
#include "utils.h"
#include <cuda.h>
#include <stdio.h>

namespace raft {
namespace spatial {
namespace knn {
namespace detail {

inline void getCountsWrap(uint32_t* counts, charMatrix ir, intMatrix sums)
{
  dim3 block(BLOCK_SIZE, 1);
  dim3 grid;
  grid.y = 1;
  uint32_t todo, numDone;

  numDone = 0;
  while (numDone < ir.pr) {
    todo   = MIN(ir.pr - numDone, MAX_BS * BLOCK_SIZE);
    grid.x = todo / BLOCK_SIZE;
    getCountsKernel<<<grid, block>>>(counts, numDone, ir, sums);
    numDone += todo;
  }
}

inline void buildMapWrap(intMatrix map, charMatrix ir, intMatrix sums, uint32_t offSet)
{
  uint32_t numScans = (ir.c + SCAN_WIDTH - 1) / SCAN_WIDTH;
  dim3 block(SCAN_WIDTH / 2, 1);
  dim3 grid;
  uint32_t todo, numDone;

  grid.x  = numScans;
  numDone = 0;
  while (numDone < ir.r) {
    todo   = MIN(ir.r - numDone, MAX_BS);
    grid.y = todo;
    buildMapKernel<<<grid, block>>>(map, ir, sums, offSet + numDone);
    numDone += todo;
  }
}

inline void sumWrap(charMatrix in, intMatrix sum)
{
  uint i;
  uint32_t todo, numDone, temp;
  uint32_t n        = in.c;
  uint32_t numScans = (n + SCAN_WIDTH - 1) / SCAN_WIDTH;
  uint32_t depth    = ceil(log(n) / log(SCAN_WIDTH)) - 1;
  uint32_t* width   = (uint32_t*)calloc(depth + 1, sizeof(*width));

  std::cout << "numScans=" << numScans << ", depth=" << depth << "width=" << width << std::endl;

  intMatrix* dAux;
  dAux = (intMatrix*)calloc(depth + 1, sizeof(*dAux));

  for (i = 0, temp = n; i <= depth; i++) {
    temp      = (temp + SCAN_WIDTH - 1) / SCAN_WIDTH;
    dAux[i].r = dAux[i].pr = in.r;
    dAux[i].c = dAux[i].pc = dAux[i].ld = temp;
    checkErr(cudaMalloc((void**)&dAux[i].mat, dAux[i].pr * dAux[i].pc * sizeof(*dAux[i].mat)));
  }

  std::cout << "Yep" << std::endl;

  dim3 block(SCAN_WIDTH / 2, 1);
  dim3 grid;

  numDone = 0;
  while (numDone < in.r) {
    todo      = MIN(in.r - numDone, MAX_BS);
    numScans  = (n + SCAN_WIDTH - 1) / SCAN_WIDTH;
    dAux[0].r = dAux[0].pr = todo;
    grid.x                 = numScans;
    grid.y                 = todo;
    sumKernel<<<grid, block>>>(in, sum, dAux[0], n);
    RAFT_CUDA_TRY(cudaDeviceSynchronize());

    width[0] = numScans;  // Necessary because following loop might not be entered
    for (i = 0; i < depth; i++) {
      width[i]      = numScans;
      numScans      = (numScans + SCAN_WIDTH - 1) / SCAN_WIDTH;
      dAux[i + 1].r = dAux[i + 1].pr = todo;

      grid.x = numScans;
      sumKernelI<<<grid, block>>>(dAux[i], dAux[i], dAux[i + 1], width[i]);
        RAFT_CUDA_TRY(cudaDeviceSynchronize());
    }

    for (i = depth - 1; i > 0; i--) {
      grid.x = width[i];
      combineSumKernel<<<grid, block>>>(dAux[i - 1], numDone, dAux[i], width[i - 1]);
        RAFT_CUDA_TRY(cudaDeviceSynchronize());
    }

    grid.x = width[0];
    combineSumKernel<<<grid, block>>>(sum, numDone, dAux[0], n);
      RAFT_CUDA_TRY(cudaDeviceSynchronize());

    numDone += todo;

    std::cout << "numDone = " << numDone << std::endl;
   }

  for (i = 0; i <= depth; i++)
    cudaFree(dAux[i].mat);
  free(dAux);
  free(width);
}

inline void dist1Wrap(const matrix dq, const matrix dx, matrix dD)
{
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid;

  uint32_t todoX, todoY, numDoneX, numDoneY;

  numDoneX = 0;
  while (numDoneX < dx.pr) {
    todoX    = MIN(dx.pr - numDoneX, MAX_BS * BLOCK_SIZE);
    grid.x   = todoX / BLOCK_SIZE;
    numDoneY = 0;
    while (numDoneY < dq.pr) {
      todoY  = MIN(dq.pr - numDoneY, MAX_BS * BLOCK_SIZE);
      grid.y = todoY / BLOCK_SIZE;
      dist1Kernel<<<grid, block>>>(dq, numDoneY, dx, numDoneX, dD);
      numDoneY += todoY;
    }
    numDoneX += todoX;
  }

  cudaThreadSynchronize();
}

inline void findRangeWrap(const matrix dD, float* dranges, uint32_t cntWant)
{
  dim3 block(4 * BLOCK_SIZE, BLOCK_SIZE / 4);
  dim3 grid(1, 4 * (dD.pr / BLOCK_SIZE));
  uint32_t numDone, todo;

  numDone = 0;
  while (numDone < dD.pr) {
    todo   = MIN(dD.pr - numDone, MAX_BS * BLOCK_SIZE / 4);
    grid.y = 4 * (todo / BLOCK_SIZE);
    findRangeKernel<<<grid, block>>>(dD, numDone, dranges, cntWant);
    numDone += todo;
    printf("numDone=%d\n", numDone);
  }
  cudaDeviceSynchronize();
}

inline void rangeSearchWrap(const matrix dD, const float* dranges, charMatrix dir)
{
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid;

  uint32_t todoX, todoY, numDoneX, numDoneY;

  numDoneX = 0;
  while (numDoneX < dD.pc) {
    todoX    = MIN(dD.pc - numDoneX, MAX_BS * BLOCK_SIZE);
    grid.x   = todoX / BLOCK_SIZE;
    numDoneY = 0;
    while (numDoneY < dD.pr) {
      todoY  = MIN(dD.pr - numDoneY, MAX_BS * BLOCK_SIZE);
      grid.y = todoY / BLOCK_SIZE;
      rangeSearchKernel<<<grid, block>>>(dD, numDoneX, numDoneY, dranges, dir);
      numDoneY += todoY;
    }
    numDoneX += todoX;
  }

  cudaThreadSynchronize();
}

inline void nnWrap(const matrix dq, const matrix dx, float* dMins, uint32_t* dMinIDs)
{
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid;
  uint32_t numDone, todo;

  grid.x = 1;

  numDone = 0;
  while (numDone < dq.pr) {
    todo   = MIN(dq.pr - numDone, MAX_BS * BLOCK_SIZE);
    grid.y = todo / BLOCK_SIZE;
    nnKernel<<<grid, block>>>(dq, numDone, dx, dMins, dMinIDs);
    numDone += todo;
  }
  cudaThreadSynchronize();
}

inline void knnWrap(const matrix dq, const matrix dx, matrix dMins, intMatrix dMinIDs)
{
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid;
  uint32_t numDone, todo;

  grid.x = 1;

  numDone = 0;
  while (numDone < dq.pr) {
    todo   = MIN(dq.pr - numDone, MAX_BS * BLOCK_SIZE);
    grid.y = todo / BLOCK_SIZE;
    knnKernel<<<grid, block>>>(dq, numDone, dx, dMins, dMinIDs);
    numDone += todo;
  }
  cudaThreadSynchronize();
}

inline void planNNWrap(const matrix dq,
                       const uint32_t* dqMap,
                       const matrix dx,
                       const intMatrix dxMap,
                       float* dMins,
                       uint32_t* dMinIDs,
                       compPlan dcP,
                       uint32_t compLength)
{
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid;
  uint32_t todo;

  grid.x        = 1;
  uint32_t numDone = 0;
  while (numDone < compLength) {
    todo   = MIN((compLength - numDone), MAX_BS * BLOCK_SIZE);
    grid.y = todo / BLOCK_SIZE;
    planNNKernel<<<grid, block>>>(dq, dqMap, dx, dxMap, dMins, dMinIDs, dcP, numDone);
    numDone += todo;
  }
  cudaThreadSynchronize();
}

inline void planKNNWrap(const matrix dq,  // query matrix
                        const uint32_t* dqMap,  //
                        const matrix dx,  // index matrix
                        const intMatrix dxMap,
                        matrix dMins,  //
                        intMatrix dMinIDs,
                        compPlan dcP,
                        uint32_t compLength)
{
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid;
  uint32_t todo;

  grid.x        = 1;
  uint32_t numDone = 0;

  int n_times = 0;
  while (numDone < compLength) {
    todo   = MIN((compLength - numDone), MAX_BS * BLOCK_SIZE);
    grid.y = todo / BLOCK_SIZE;
    planKNNKernel<<<grid, block>>>(dq, dqMap, dx, dxMap, dMins, dMinIDs, dcP, numDone);
    numDone += todo;
    ++n_times;
  }

  printf("n_times: %d\n", n_times);
  cudaThreadSynchronize();
}

//                         query matrix     index matrix
inline void rangeCountWrap(const matrix dq, const matrix dx, float* dranges, uint32_t* dcounts)
{
  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid;
  uint32_t numDone, todo;

  grid.x = 1;

  numDone = 0;
  while (numDone < dq.pr) {
    todo   = MIN(dq.pr - numDone, MAX_BS * BLOCK_SIZE);
    grid.y = todo / BLOCK_SIZE;
    rangeCountKernel<<<grid, block>>>(dq, numDone, dx, dranges, dcounts);
    numDone += todo;
  }
  cudaThreadSynchronize();
}

}  // namespace detail
}  // namespace knn
}  // namespace spatial
};  // namespace raft
#endif
