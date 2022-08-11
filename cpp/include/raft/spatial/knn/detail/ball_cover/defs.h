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

#ifndef DEFS_H
#define DEFS_H

#include <float.h>
#include <stdint.h>

#define FLOAT_TOL 1e-7

#define KMAX       32  // Internal parameter.  Do not change!
#define BLOCK_SIZE 16  // must be a power of 2 (current
// implementation of findRange requires a power of 4, in fact)

#define MAX_BS     65535  // max block size (specified by CUDA)
#define SCAN_WIDTH 1024

#define MEM_USED_IN_SCAN(n) (2 * ((n) + SCAN_WIDTH - 1) / SCAN_WIDTH * sizeof(unint))

// The distance measure that is used.  This macro returns the
// distance for a single coordinate.
//#define DIST(i,j) ( fabs((i)-(j)) )  // L_1
#define DIST(i, j) (((i) - (j)) * ((i) - (j)))  // L_2

// Format that the data is manipulated in:
typedef float real;
typedef uint32_t unint;

#define MAX_REAL FLT_MAX

// To switch to double precision, comment out the above
// 2 lines and uncomment the following two lines.

// typedef double real;
//#define MAX_REAL DBL_MAX

// Percentage of device mem to use
#define MEM_USABLE .95

#define DUMMY_IDX UINT_MAX

// Row major indexing
#define IDX(i, j, ld) (((size_t)(i) * (ld)) + (j))

// increase an int to the next multiple of BLOCK_SIZE
#define PAD(i) (((i) % BLOCK_SIZE) == 0 ? (i) : ((i) / BLOCK_SIZE) * BLOCK_SIZE + BLOCK_SIZE)

// decrease an int to the next multiple of BLOCK_SIZE
#define DPAD(i) (((i) % BLOCK_SIZE) == 0 ? (i) : ((i) / BLOCK_SIZE) * BLOCK_SIZE)

#define MAX(i, j)        ((i) > (j) ? (i) : (j))
#define MIN(i, j)        ((i) <= (j) ? (i) : (j))
#define MAXi(i, j, k, l) ((i) > (j) ? (k) : (l))  // indexed version
#define MINi(i, j, k, l) ((i) <= (j) ? (k) : (l))

namespace raft {
namespace spatial {
namespace knn {
namespace detail {

typedef struct {
  real* mat;
  unint r;   // rows
  unint c;   // cols
  unint pr;  // padded rows
  unint pc;  // padded cols
  unint ld;  // the leading dimension (in this code, this is the same as pc)
} matrix;

typedef struct {
  char* mat;
  unint r;
  unint c;
  unint pr;
  unint pc;
  unint ld;
} charMatrix;

typedef struct {
  unint* mat;
  unint r;
  unint c;
  unint pr;
  unint pc;
  unint ld;
} intMatrix;

typedef struct {
  // TODO: Number of landmarks?
  unint* numGroups;  // The number of groups of DB points to be examined.

  // TODO: Number of elements in each landmark?
  unint* groupCountX;  // The number of elements in each DB group.

  // Inverted index from each query to each query group
  unint* qToQGroup;  // map from query to query group #.

  // Map from query landmark to each index landmark
  unint* qGroupToXGroup;  // map from query group to DB gruop

  //
  unint ld;  // the width of memPos and groupCount (= max over numGroups)
} compPlan;

typedef struct {
  //
  matrix dx;
  intMatrix dxMap;
  matrix dr;
  unint* groupCount;
} rbcStruct;

}  // namespace detail
}  // namespace knn
}  // namespace spatial
};  // namespace raft
#endif
