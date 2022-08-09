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

#ifndef RBC_CU
#define RBC_CU

#include "defs.h"
#include "kernel_wrap.cuh"
#include "kernels.cuh"
#include "utils.h"
#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>

namespace raft {
namespace spatial {
namespace knn {
namespace detail {

// Assign each point in dq to its nearest point in dr.
inline void computeReps(matrix dq, matrix dr, uint32_t* repIDs, float* distToReps)
{
  float* dMins;
  uint32_t* dMinIDs;

  checkErr(cudaMalloc((void**)&(dMins), dq.pr * sizeof(*dMins)));
  checkErr(cudaMalloc((void**)&(dMinIDs), dq.pr * sizeof(*dMinIDs)));

  nnWrap(dq, dr, dMins, dMinIDs);

  cudaMemcpy(distToReps, dMins, dq.r * sizeof(*dMins), cudaMemcpyDeviceToHost);
  cudaMemcpy(repIDs, dMinIDs, dq.r * sizeof(*dMinIDs), cudaMemcpyDeviceToHost);

  cudaFree(dMins);
  cudaFree(dMinIDs);
}

// Assumes radii is initialized to 0s
inline void computeRadii(uint32_t* repIDs, float* distToReps, float* radii, uint32_t n, uint32_t numReps)
{
  uint32_t i;

  for (i = 0; i < n; i++)
    radii[repIDs[i]] = MAX(distToReps[i], radii[repIDs[i]]);
}

// Assumes groupCount is initialized to 0s
inline void computeCounts(uint32_t* repIDs, uint32_t n, uint32_t* groupCount)
{
  uint32_t i;

  for (i = 0; i < n; i++) {
    groupCount[repIDs[i]]++;
  }

//  printf("Done counts\n");
}

//                    query
inline void buildQMap(matrix q, uint32_t* qMap, uint32_t* repIDs, uint32_t numReps, uint32_t* compLength)
{
  uint32_t n = q.r;
  uint32_t i;
  uint32_t* gS;  // groupSize

  gS = (uint32_t*)calloc(numReps + 1, sizeof(*gS));

  for (i = 0; i < n; i++)
    gS[repIDs[i] + 1]++;
  for (i = 0; i < numReps + 1; i++)
    gS[i] = PAD(gS[i]);

  for (i = 1; i < numReps + 1; i++)
    gS[i] = gS[i - 1] + gS[i];

  *compLength = gS[numReps];

  for (i = 0; i < (*compLength); i++)
    qMap[i] = DUMMY_IDX;

  for (i = 0; i < n; i++) {
    qMap[gS[repIDs[i]]] = i;
    gS[repIDs[i]]++;
  }

  free(gS);
}

// Sets the computation matrix to the identity.
inline void idIntersection(charMatrix cM)
{
  uint32_t i;
  for (i = 0; i < cM.r; i++) {
    if (i < cM.c) cM.mat[IDX(i, i, cM.ld)] = 1;
  }
}

inline void fullIntersection(charMatrix cM)
{
  uint32_t i, j;
  for (i = 0; i < cM.r; i++) {
    for (j = 0; j < cM.c; j++) {
      cM.mat[IDX(i, j, cM.ld)] = 1;
    }
  }
}

// Choose representatives and move them to device
inline void setupReps(matrix x, rbcStruct* rbcS, uint32_t numReps)
{
  uint32_t i;
  uint32_t* randInds;
  randInds = (uint32_t*)calloc(PAD(numReps), sizeof(*randInds));
  subRandPerm(numReps, x.r, randInds);

  matrix r;
  r.r  = numReps;
  r.pr = PAD(numReps);
  r.c  = x.c;
  r.pc = r.ld = PAD(r.c);
  r.mat       = (float*)calloc(r.pr * r.pc, sizeof(*r.mat));

  for (i = 0; i < numReps; i++)
    copyVector(&r.mat[IDX(i, 0, r.ld)], &x.mat[IDX(randInds[i], 0, x.ld)], x.c);

  copyAndMove(&rbcS->dr, &r);

  free(randInds);
  free(r.mat);
}

inline void computeKNNs(matrix dx,          // index matrix
                        intMatrix dxMap,
                        matrix dq,          // query matrix
                        uint32_t* dqMap,
                        compPlan dcP,
                        intMatrix NNs,
                        matrix NNdists,
                        uint32_t compLength)
{
  matrix dNNdists;
  intMatrix dMinIDs;
  dNNdists.r  = compLength;
  dNNdists.pr = compLength;
  dNNdists.c  = KMAX;
  dNNdists.pc = KMAX;
  dNNdists.ld = dNNdists.pc;
  dMinIDs.r   = compLength;
  dMinIDs.pr  = compLength;
  dMinIDs.c   = KMAX;
  dMinIDs.pc  = KMAX;
  dMinIDs.ld  = dMinIDs.pc;

  checkErr(cudaMalloc((void**)&dNNdists.mat, dNNdists.pr * dNNdists.pc * sizeof(*dNNdists.mat)));
  checkErr(cudaMalloc((void**)&dMinIDs.mat, dMinIDs.pr * dMinIDs.pc * sizeof(*dMinIDs.mat)));

  planKNNWrap(dq, dqMap, dx, dxMap, dNNdists, dMinIDs, dcP, compLength);
  cudaMemcpy(NNs.mat, dMinIDs.mat, dq.r * KMAX * sizeof(*NNs.mat), cudaMemcpyDeviceToHost);
  cudaMemcpy(NNdists.mat, dNNdists.mat, dq.r * KMAX * sizeof(*NNdists.mat), cudaMemcpyDeviceToHost);

  cudaFree(dNNdists.mat);
  cudaFree(dMinIDs.mat);
}

// This calls the dist1Kernel wrapper, but has it compute only
// a submatrix of the all-pairs distance matrix.  In particular,
// only distances from dr[start,:].. dr[start+length-1] to all of x
// are computed, resulting in a distance matrix of size
// length by dx.pr.  It is assumed that length is padded.
inline void distSubMat(matrix dr, matrix dx, matrix dD, uint32_t start, uint32_t length)
{
  dr.r = dr.pr = length;
  dr.mat       = &dr.mat[IDX(start, 0, dr.ld)];
  dist1Wrap(dr, dx, dD);
}

inline void destroyRBC(rbcStruct* rbcS)
{
  cudaFree(rbcS->dx.mat);
  cudaFree(rbcS->dxMap.mat);
  cudaFree(rbcS->dr.mat);
  free(rbcS->groupCount);
}

/* Danger: this function allocates memory that it does not free.
 * Use freeCompPlan to clear mem.
 * See the readme.txt file for a description of why this function is needed.
 */
inline void initCompPlan(
  compPlan* dcP, charMatrix cM, uint32_t* groupCountQ, uint32_t* groupCountX, uint32_t numReps)
{
  uint32_t i, j, k;
  uint32_t maxNumGroups = 0;
  compPlan cP;

  uint32_t sNumGroups = numReps;
  cP.numGroups     = (uint32_t*)calloc(sNumGroups, sizeof(*cP.numGroups));

  for (i = 0; i < numReps; i++) {
    cP.numGroups[i] = 0;
    for (j = 0; j < numReps; j++)
      cP.numGroups[i] += cM.mat[IDX(i, j, cM.ld)];
    maxNumGroups = MAX(cP.numGroups[i], maxNumGroups);
  }
  cP.ld = maxNumGroups;

  uint32_t sQToQGroup;
  for (i = 0, sQToQGroup = 0; i < numReps; i++)
    sQToQGroup += PAD(groupCountQ[i]);

  cP.qToQGroup = (uint32_t*)calloc(sQToQGroup, sizeof(*cP.qToQGroup));

  for (i = 0, k = 0; i < numReps; i++) {
    for (j = 0; j < PAD(groupCountQ[i]); j++)
      cP.qToQGroup[k++] = i;
  }

  uint32_t sQGroupToXGroup = numReps * maxNumGroups;
  cP.qGroupToXGroup     = (uint32_t*)calloc(sQGroupToXGroup, sizeof(*cP.qGroupToXGroup));
  uint32_t sGroupCountX    = maxNumGroups * numReps;
  cP.groupCountX        = (uint32_t*)calloc(sGroupCountX, sizeof(*cP.groupCountX));

  for (i = 0; i < numReps; i++) {
    for (j = 0, k = 0; j < numReps; j++) {
      if (cM.mat[IDX(i, j, cM.ld)]) {
        cP.qGroupToXGroup[IDX(i, k, cP.ld)] = j;
        cP.groupCountX[IDX(i, k++, cP.ld)]  = groupCountX[j];
      }
    }
  }

  // Move to device
  checkErr(cudaMalloc((void**)&dcP->numGroups, sNumGroups * sizeof(*dcP->numGroups)));
  cudaMemcpy(
    dcP->numGroups, cP.numGroups, sNumGroups * sizeof(*dcP->numGroups), cudaMemcpyHostToDevice);
  checkErr(cudaMalloc((void**)&dcP->groupCountX, sGroupCountX * sizeof(*dcP->groupCountX)));
  cudaMemcpy(dcP->groupCountX,
             cP.groupCountX,
             sGroupCountX * sizeof(*dcP->groupCountX),
             cudaMemcpyHostToDevice);
  checkErr(cudaMalloc((void**)&dcP->qToQGroup, sQToQGroup * sizeof(*dcP->qToQGroup)));
  cudaMemcpy(
    dcP->qToQGroup, cP.qToQGroup, sQToQGroup * sizeof(*dcP->qToQGroup), cudaMemcpyHostToDevice);
  checkErr(
    cudaMalloc((void**)&dcP->qGroupToXGroup, sQGroupToXGroup * sizeof(*dcP->qGroupToXGroup)));
  cudaMemcpy(dcP->qGroupToXGroup,
             cP.qGroupToXGroup,
             sQGroupToXGroup * sizeof(*dcP->qGroupToXGroup),
             cudaMemcpyHostToDevice);
  dcP->ld = cP.ld;

  free(cP.numGroups);
  free(cP.groupCountX);
  free(cP.qToQGroup);
  free(cP.qGroupToXGroup);
}

// Frees memory allocated in initCompPlan.
inline void freeCompPlan(compPlan* dcP)
{
  cudaFree(dcP->numGroups);
  cudaFree(dcP->groupCountX);
  cudaFree(dcP->qToQGroup);
  cudaFree(dcP->qGroupToXGroup);
}

// This function is very similar to queryRBC, with a couple of basic changes to handle
// k-nn.
inline void kqueryRBC(const matrix q, const rbcStruct rbcS, intMatrix NNs, matrix NNdists)
{
  uint32_t m = q.r;

  // number of rows
  uint32_t numReps = rbcS.dr.r;

  uint32_t compLength;

//  printf("Create comp plan\n");
  compPlan dcP;
  uint32_t *qMap, *dqMap;
  qMap = (uint32_t*)calloc(PAD(m + (BLOCK_SIZE - 1) * PAD(numReps)), sizeof(*qMap));
  matrix dq;
  copyAndMove(&dq, &q);

//  printf("Create char matrix\n");
  charMatrix cM;
  cM.r = cM.c = numReps;
  cM.pr = cM.pc = cM.ld = PAD(numReps);
  cM.mat                = (char*)calloc(cM.pr * cM.pc, sizeof(*cM.mat));

  uint32_t* repIDsQ;
  repIDsQ = (uint32_t*)calloc(m, sizeof(*repIDsQ));
  float* distToRepsQ;
  distToRepsQ = (float*)calloc(m, sizeof(*distToRepsQ));
  uint32_t* groupCountQ;
  groupCountQ = (uint32_t*)calloc(PAD(numReps), sizeof(*groupCountQ));

  // Assign each point in dq to its nearest point in dr.
  printf("Compute reps\n");

  computeReps(dq, rbcS.dr, repIDsQ, distToRepsQ);

  printf("Compute counts\n");
  // How many points are assigned to each group?
  computeCounts(repIDsQ, m, groupCountQ);

  printf("Build qmap\n");
  // Set up the mapping from groups to queries (qMap).
  buildQMap(q, qMap, repIDsQ, numReps, &compLength);

  // Setup the computation matrix.  Currently, the computation matrix is
  // just the identity matrix: each query assigned to a particular
  // representative is compared only to that representative's points.

  // NOTE: currently, idIntersection is the *only* computation matrix
  // that will work properly with k-nn search (this is not true for 1-nn above).
  printf("Id intersection\n");
  idIntersection(cM);

  printf("init comp plan\n");
  initCompPlan(&dcP, cM, groupCountQ, rbcS.groupCount, numReps);

  checkErr(cudaMalloc((void**)&dqMap, compLength * sizeof(*dqMap)));
  cudaMemcpy(dqMap, qMap, compLength * sizeof(*dqMap), cudaMemcpyHostToDevice);

  printf("Compute knns\n");
  computeKNNs(rbcS.dx, rbcS.dxMap, dq, dqMap, dcP, NNs, NNdists, compLength);

  free(qMap);
  freeCompPlan(&dcP);
  cudaFree(dq.mat);
  free(cM.mat);
  free(repIDsQ);
  free(distToRepsQ);
  free(groupCountQ);
}

inline void buildRBC(const matrix x, rbcStruct* rbcS, uint32_t numReps, uint32_t s)
{
  uint32_t n = x.pr;
  intMatrix xmap;

  printf("Setting up reps\n");
  setupReps(x, rbcS, numReps);
  copyAndMove(&rbcS->dx, &x);

  printf("After copy and move\n");

  xmap.r  = numReps;
  xmap.pr = PAD(numReps);
  xmap.c  = s;
  xmap.pc = xmap.ld = PAD(s);
  xmap.mat          = (uint32_t*)calloc(xmap.pr * xmap.pc, sizeof(*xmap.mat));
  copyAndMoveI(&rbcS->dxMap, &xmap);

  printf("numReps: %u\n", numReps);
  rbcS->groupCount = (uint32_t*)calloc(PAD(numReps), sizeof(*rbcS->groupCount));

  printf("After copy and move 2\n");

  // Figure out how much fits into memory
  size_t memFree, memTot;
  cudaMemGetInfo(&memFree, &memTot);
  memFree = (uint32_t)(((float)memFree) * MEM_USABLE);

  printf("After get info\n");
  /* mem needed per rep:
   *  n*sizeof(float) - dist mat
   *  n*sizeof(char) - dir
   *  n*sizeof(int)  - dSums
   *  sizeof(float)   - dranges
   *  sizeof(int)    - dCnts
   *  MEM_USED_IN_SCAN - memory used internally
   */
  uint32_t ptsAtOnce = DPAD(memFree / ((n + 1) * sizeof(float) + n * sizeof(char) +
                                    (n + 1) * sizeof(uint32_t) + 2 * MEM_USED_IN_SCAN(n)));
  if (!ptsAtOnce) {
    fprintf(stderr,
            "error: %lu is not enough memory to build the RBC.. exiting\n",
            (unsigned long)memFree);
    exit(1);
  }

  printf("After pdad\n");

  // Now set everything up for the scans
  matrix dD;
  dD.pr = dD.r = ptsAtOnce;   // number of rows
  dD.c         = rbcS->dx.r;  // col
  dD.pc        = rbcS->dx.pr;
  dD.ld        = dD.pc;

  checkErr(cudaMalloc((void**)&dD.mat, dD.pr * dD.pc * sizeof(*dD.mat)));

  float* dranges;
  checkErr(cudaMalloc((void**)&dranges, ptsAtOnce * sizeof(float)));

  charMatrix ir;
  ir.r   = dD.r;
  ir.pr  = dD.pr;
  ir.c   = dD.c;
  ir.pc  = dD.pc;
  ir.ld  = dD.ld;
  ir.mat = (char*)calloc(ir.pr * ir.pc, sizeof(*ir.mat));
  charMatrix dir;
  copyAndMoveC(&dir, &ir);

  intMatrix dSums;  // used to compute memory addresses.
  dSums.r  = dir.r;
  dSums.pr = dir.pr;
  dSums.c  = dir.c;
  dSums.pc = dir.pc;
  dSums.ld = dir.ld;
  checkErr(cudaMalloc((void**)&dSums.mat, dSums.pc * dSums.pr * sizeof(*dSums.mat)));

  uint32_t* dCnts;
  checkErr(cudaMalloc((void**)&dCnts, ptsAtOnce * sizeof(*dCnts)));

  // Do the scans to build the dxMap
  uint32_t numLeft = rbcS->dr.r;  // points left to process
  uint32_t row     = 0;           // base row for iteration of while loop
  uint32_t pi, pip;               // pi=pts per it, pip=pad(pi)

  printf("About to start while loop\n");

  while (numLeft > 0) {
    pi       = MIN(ptsAtOnce, numLeft);  // points to do this iteration.
    pip      = PAD(pi);
    dD.r     = pi;
    dD.pr    = pip;
    dir.r    = pi;
    dir.pr   = pip;
    dSums.r  = pi;
    dSums.pr = pip;

    printf("distSubMat\n");

    distSubMat(rbcS->dr, rbcS->dx, dD, row, pip);  // compute the distance matrix

    printf("findRangeWrap\n");
    findRangeWrap(dD, dranges, s);  // find an appropriate range

    printf("Range search\n");
    rangeSearchWrap(dD, dranges, dir);  // set binary vector for points in range

    printf("sumWrap\n");
    sumWrap(dir, dSums);  // This and the next call perform the parallel compaction.

    printf("buildMapWrap\n");
    buildMapWrap(rbcS->dxMap, dir, dSums, row);

    printf("getCountsWrap\n");
    getCountsWrap(dCnts, dir, dSums);  // How many points are assigned to each rep?  It is not
    //*exactly* s, which is why we need to compute this.

    printf("cudaMemCpy\n");
    cudaMemcpy(
      &rbcS->groupCount[row], dCnts, pi * sizeof(*rbcS->groupCount), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    numLeft -= pi;
    row += pi;
    printf("num_left=%d\n", numLeft);
  }

//  printf("Done.\n");

  cudaFree(dCnts);
  free(ir.mat);
  free(xmap.mat);
  cudaFree(dranges);
  cudaFree(dir.mat);
  cudaFree(dSums.mat);
  cudaFree(dD.mat);
}

}  // namespace detail
}  // namespace knn
}  // namespace spatial
};  // namespace raft

#endif
