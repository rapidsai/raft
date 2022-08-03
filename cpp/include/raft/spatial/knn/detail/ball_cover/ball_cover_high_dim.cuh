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
inline void computeReps(matrix dq, matrix dr, unint* repIDs, real* distToReps)
{
  real* dMins;
  unint* dMinIDs;

  checkErr(cudaMalloc((void**)&(dMins), dq.pr * sizeof(*dMins)));
  checkErr(cudaMalloc((void**)&(dMinIDs), dq.pr * sizeof(*dMinIDs)));

  nnWrap(dq, dr, dMins, dMinIDs);

  cudaMemcpy(distToReps, dMins, dq.r * sizeof(*dMins), cudaMemcpyDeviceToHost);
  cudaMemcpy(repIDs, dMinIDs, dq.r * sizeof(*dMinIDs), cudaMemcpyDeviceToHost);

  cudaFree(dMins);
  cudaFree(dMinIDs);
}

// Assumes radii is initialized to 0s
inline void computeRadii(unint* repIDs, real* distToReps, real* radii, unint n, unint numReps)
{
  unint i;

  for (i = 0; i < n; i++)
    radii[repIDs[i]] = MAX(distToReps[i], radii[repIDs[i]]);
}

// Assumes groupCount is initialized to 0s
inline void computeCounts(unint* repIDs, unint n, unint* groupCount)
{
  unint i;

  for (i = 0; i < n; i++) {
    groupCount[repIDs[i]]++;
  }

  printf("Done counts\n");
}

inline void buildQMap(matrix q, unint* qMap, unint* repIDs, unint numReps, unint* compLength)
{
  unint n = q.r;
  unint i;
  unint* gS;  // groupSize

  gS = (unint*)calloc(numReps + 1, sizeof(*gS));

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
  unint i;
  for (i = 0; i < cM.r; i++) {
    if (i < cM.c) cM.mat[IDX(i, i, cM.ld)] = 1;
  }
}

inline void fullIntersection(charMatrix cM)
{
  unint i, j;
  for (i = 0; i < cM.r; i++) {
    for (j = 0; j < cM.c; j++) {
      cM.mat[IDX(i, j, cM.ld)] = 1;
    }
  }
}

// Choose representatives and move them to device
inline void setupReps(matrix x, rbcStruct* rbcS, unint numReps)
{
  unint i;
  unint* randInds;
  randInds = (unint*)calloc(PAD(numReps), sizeof(*randInds));
  subRandPerm(numReps, x.r, randInds);

  matrix r;
  r.r  = numReps;
  r.pr = PAD(numReps);
  r.c  = x.c;
  r.pc = r.ld = PAD(r.c);
  r.mat       = (real*)calloc(r.pr * r.pc, sizeof(*r.mat));

  for (i = 0; i < numReps; i++)
    copyVector(&r.mat[IDX(i, 0, r.ld)], &x.mat[IDX(randInds[i], 0, x.ld)], x.c);

  copyAndMove(&rbcS->dr, &r);

  free(randInds);
  free(r.mat);
}

inline void computeKNNs(matrix dx,
                        intMatrix dxMap,
                        matrix dq,
                        unint* dqMap,
                        compPlan dcP,
                        intMatrix NNs,
                        matrix NNdists,
                        unint compLength)
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
inline void distSubMat(matrix dr, matrix dx, matrix dD, unint start, unint length)
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
  compPlan* dcP, charMatrix cM, unint* groupCountQ, unint* groupCountX, unint numReps)
{
  unint i, j, k;
  unint maxNumGroups = 0;
  compPlan cP;

  unint sNumGroups = numReps;
  cP.numGroups     = (unint*)calloc(sNumGroups, sizeof(*cP.numGroups));

  for (i = 0; i < numReps; i++) {
    cP.numGroups[i] = 0;
    for (j = 0; j < numReps; j++)
      cP.numGroups[i] += cM.mat[IDX(i, j, cM.ld)];
    maxNumGroups = MAX(cP.numGroups[i], maxNumGroups);
  }
  cP.ld = maxNumGroups;

  unint sQToQGroup;
  for (i = 0, sQToQGroup = 0; i < numReps; i++)
    sQToQGroup += PAD(groupCountQ[i]);

  cP.qToQGroup = (unint*)calloc(sQToQGroup, sizeof(*cP.qToQGroup));

  for (i = 0, k = 0; i < numReps; i++) {
    for (j = 0; j < PAD(groupCountQ[i]); j++)
      cP.qToQGroup[k++] = i;
  }

  unint sQGroupToXGroup = numReps * maxNumGroups;
  cP.qGroupToXGroup     = (unint*)calloc(sQGroupToXGroup, sizeof(*cP.qGroupToXGroup));
  unint sGroupCountX    = maxNumGroups * numReps;
  cP.groupCountX        = (unint*)calloc(sGroupCountX, sizeof(*cP.groupCountX));

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
  unint m = q.r;

  // number of rows
  unint numReps = rbcS.dr.r;

  unint compLength;

  printf("Create comp plan\n");
  compPlan dcP;
  unint *qMap, *dqMap;
  qMap = (unint*)calloc(PAD(m + (BLOCK_SIZE - 1) * PAD(numReps)), sizeof(*qMap));
  matrix dq;
  copyAndMove(&dq, &q);

  printf("Create char matrix\n");
  charMatrix cM;
  cM.r = cM.c = numReps;
  cM.pr = cM.pc = cM.ld = PAD(numReps);
  cM.mat                = (char*)calloc(cM.pr * cM.pc, sizeof(*cM.mat));

  unint* repIDsQ;
  repIDsQ = (unint*)calloc(m, sizeof(*repIDsQ));
  real* distToRepsQ;
  distToRepsQ = (real*)calloc(m, sizeof(*distToRepsQ));
  unint* groupCountQ;
  groupCountQ = (unint*)calloc(PAD(numReps), sizeof(*groupCountQ));

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

inline void buildRBC(const matrix x, rbcStruct* rbcS, unint numReps, unint s)
{
  unint n = x.pr;
  intMatrix xmap;

  printf("Setting up reps\n");
  setupReps(x, rbcS, numReps);
  copyAndMove(&rbcS->dx, &x);

  printf("After copy and move\n");

  xmap.r  = numReps;
  xmap.pr = PAD(numReps);
  xmap.c  = s;
  xmap.pc = xmap.ld = PAD(s);
  xmap.mat          = (unint*)calloc(xmap.pr * xmap.pc, sizeof(*xmap.mat));
  copyAndMoveI(&rbcS->dxMap, &xmap);

  printf("numReps: %u\n", numReps);
  rbcS->groupCount = (unint*)calloc(PAD(numReps), sizeof(*rbcS->groupCount));

  printf("After copy and move 2\n");

  // Figure out how much fits into memory
  size_t memFree, memTot;
  cudaMemGetInfo(&memFree, &memTot);
  memFree = (unint)(((float)memFree) * MEM_USABLE);

  printf("After get info\n");
  /* mem needed per rep:
   *  n*sizeof(real) - dist mat
   *  n*sizeof(char) - dir
   *  n*sizeof(int)  - dSums
   *  sizeof(real)   - dranges
   *  sizeof(int)    - dCnts
   *  MEM_USED_IN_SCAN - memory used internally
   */
  unint ptsAtOnce = DPAD(memFree / ((n + 1) * sizeof(real) + n * sizeof(char) +
                                    (n + 1) * sizeof(unint) + 2 * MEM_USED_IN_SCAN(n)));
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

  real* dranges;
  checkErr(cudaMalloc((void**)&dranges, ptsAtOnce * sizeof(real)));

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

  unint* dCnts;
  checkErr(cudaMalloc((void**)&dCnts, ptsAtOnce * sizeof(*dCnts)));

  // Do the scans to build the dxMap
  unint numLeft = rbcS->dr.r;  // points left to process
  unint row     = 0;           // base row for iteration of while loop
  unint pi, pip;               // pi=pts per it, pip=pad(pi)

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

    numLeft -= pi;
    row += pi;
    printf("num_left=%d\n", numLeft);
  }

  printf("Done.\n");

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
