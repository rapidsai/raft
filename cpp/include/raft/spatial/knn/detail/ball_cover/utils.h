/* This file is part of the Random Ball Cover (RBC) library.
 * (C) Copyright 2010, Lawrence Cayton [lcayton@tuebingen.mpg.de]
 */

#ifndef UTILS_CU
#define UTILS_CU

#include "defs.h"
#include "kernels.cuh"
#include <stdio.h>
#include <sys/time.h>

namespace raft {
namespace spatial {
namespace knn {
namespace detail {

inline void checkErr(cudaError_t cError)
{
  if (cError != cudaSuccess) {
    fprintf(stderr, "GPU-related error:\n\t%s \n", cudaGetErrorString(cError));
    fprintf(stderr, "exiting ..\n");
    exit(1);
  }
}

inline void checkErr(char* loc, cudaError_t cError)
{
  printf("in %s:\n", loc);
  checkErr(cError);
}

inline void swap(unint* a, unint* b)
{
  unint t;
  t  = *a;
  *a = *b;
  *b = t;
}

// generates a rand int in rand [a,b)
inline unint randBetween(unint a, unint b)
{
  unint val, c;

  if (b <= a) {
    fprintf(stderr, "misuse of randBetween.. exiting\n");
    exit(1);
  }
  c = b - a;

  while (c <= (val = rand() / (int)(((unsigned)RAND_MAX + 1) / c)))
    ;
  val = val + a;

  return val;
}

inline void printMat(matrix A)
{
  unint i, j;
  for (i = 0; i < A.r; i++) {
    for (j = 0; j < A.c; j++)
      printf("%6.4f ", (float)A.mat[IDX(i, j, A.ld)]);
    printf("\n");
  }
}

inline void printMatWithIDs(matrix A, unint* id)
{
  unint i, j;
  for (i = 0; i < A.r; i++) {
    for (j = 0; j < A.c; j++)
      printf("%6.4f ", (float)A.mat[IDX(i, j, A.ld)]);
    printf("%d ", id[i]);
    printf("\n");
  }
}

inline void printCharMat(charMatrix A)
{
  unint i, j;
  for (i = 0; i < A.r; i++) {
    for (j = 0; j < A.c; j++)
      printf("%d ", (char)A.mat[IDX(i, j, A.ld)]);
    printf("\n");
  }
}

inline void printIntMat(intMatrix A)
{
  unint i, j;
  for (i = 0; i < A.r; i++) {
    for (j = 0; j < A.c; j++)
      printf("%d ", (unint)A.mat[IDX(i, j, A.ld)]);
    printf("\n");
  }
}

inline void printVector(real* x, unint d)
{
  unint i;

  for (i = 0; i < d; i++)
    printf("%6.2f ", x[i]);
  printf("\n");
}

inline void copyVector(real* x, real* y, unint d)
{
  unint i;

  for (i = 0; i < d; i++)
    x[i] = y[i];
}

inline void copyMat(matrix* x, matrix* y)
{
  unint i, j;

  x->r  = y->r;
  x->pr = y->pr;
  x->c  = y->c;
  x->pc = y->pc;
  x->ld = y->ld;
  for (i = 0; i < y->r; i++) {
    for (j = 0; j < y->c; j++) {
      x->mat[IDX(i, j, x->ld)] = y->mat[IDX(i, j, y->ld)];
    }
  }
}

inline void initMat(matrix* x, unint r, unint c)
{
  x->r  = r;
  x->c  = c;
  x->pr = PAD(r);
  x->pc = PAD(c);
  x->ld = PAD(c);
}

void initIntMat(intMatrix* x, unint r, unint c)
{
  x->r  = r;
  x->c  = c;
  x->pr = PAD(r);
  x->pc = PAD(c);
  x->ld = PAD(c);
}

// returns the size of a matrix in bytes
inline size_t sizeOfMatB(matrix x) { return ((size_t)x.pr) * x.pc * sizeof(*x.mat); }

inline size_t sizeOfIntMatB(intMatrix x) { return ((size_t)x.pr) * x.pc * sizeof(*x.mat); }

// returns the numbers of elements in a matrix
inline size_t sizeOfMat(matrix x) { return ((size_t)x.pr) * x.pc; }

inline size_t sizeOfIntMat(intMatrix x) { return ((size_t)x.pr) * x.pc; }

inline real distVec(matrix x, matrix y, unint k, unint l)
{
  unint i;
  real ans = 0;

  for (i = 0; i < x.c; i++)
    ans += DIST(x.mat[IDX(k, i, x.ld)], y.mat[IDX(l, i, x.ld)]);
  // ans+=fabs(x.mat[IDX(k,i,x.ld)]-y.mat[IDX(l,i,x.ld)]);
  return ans;
}

inline double timeDiff(struct timeval start, struct timeval end)
{
  return (double)(end.tv_sec + end.tv_usec / 1e6 - start.tv_sec - start.tv_usec / 1e6);
}

inline void copyAndMove(matrix* dx, const matrix* x)
{
  dx->r  = x->r;
  dx->c  = x->c;
  dx->pr = x->pr;
  dx->pc = x->pc;
  dx->ld = x->ld;

  checkErr(cudaMalloc((void**)&(dx->mat), dx->pr * dx->pc * sizeof(*(dx->mat))));
  cudaMemcpy(dx->mat, x->mat, dx->pr * dx->pc * sizeof(*(dx->mat)), cudaMemcpyHostToDevice);
}

inline void copyAndMoveI(intMatrix* dx, const intMatrix* x)
{
  dx->r  = x->r;
  dx->c  = x->c;
  dx->pr = x->pr;
  dx->pc = x->pc;
  dx->ld = x->ld;

  checkErr(cudaMalloc((void**)&(dx->mat), dx->pr * dx->pc * sizeof(*(dx->mat))));
  cudaMemcpy(dx->mat, x->mat, dx->pr * dx->pc * sizeof(*(dx->mat)), cudaMemcpyHostToDevice);
}

inline void copyAndMoveC(charMatrix* dx, const charMatrix* x)
{
  dx->r  = x->r;
  dx->c  = x->c;
  dx->pr = x->pr;
  dx->pc = x->pc;
  dx->ld = x->ld;

  checkErr(cudaMalloc((void**)&(dx->mat), dx->pr * dx->pc * sizeof(*(dx->mat))));
  cudaMemcpy(dx->mat, x->mat, dx->pr * dx->pc * sizeof(*(dx->mat)), cudaMemcpyHostToDevice);
}

// Returns a length l subset of a random permutation of [0,...,n-1]
// using the knuth shuffle.
// Input variable x is assumed to be alloced and of size l.
void subRandPerm(unint l, unint n, unint* x)
{
  unint i, ri, *y;
  y = (unint*)calloc(n, sizeof(*y));

  struct timeval t3;
  gettimeofday(&t3, NULL);
  srand(t3.tv_usec);

  for (i = 0; i < n; i++)
    y[i] = i;

  for (i = 0; i < MIN(l, n - 1); i++) {  // The n-1 bit is necessary because you can't swap the last
    // element with something larger.
    ri = randBetween(i + 1, n);
    swap(&y[i], &y[ri]);
  }

  for (i = 0; i < l; i++)
    x[i] = y[i];
  free(y);
}

// Generates a random permutation of 0, ... , n-1 using the knuth shuffle.
// This should probably be merged with subRandPerm.
inline void randPerm(unint n, unint* x)
{
  unint i, ri;

  struct timeval t3;
  gettimeofday(&t3, NULL);
  srand(t3.tv_usec);

  for (i = 0; i < n; i++) {
    x[i] = i;
  }

  for (i = 0; i < n - 1; i++) {
    ri = randBetween(i + 1, n);
    swap(&x[i], &x[ri]);
  }
}

}  // namespace detail
}  // namespace knn
}  // namespace spatial
};  // namespace raft

#endif
