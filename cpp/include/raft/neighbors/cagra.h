/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <dlpack/dlpack.h>
#include <raft/core/c_api.h>

#ifdef __cplusplus
extern "C" {
#endif

enum cagraGraphBuildAlgo {
  /* Use IVF-PQ to build all-neighbors knn graph */
  IVF_PQ,
  /* Experimental, use NN-Descent to build all-neighbors knn graph */
  NN_DESCENT
};

struct cagraIndexParams {
  /** Degree of input graph for pruning. */
  size_t intermediate_graph_degree = 128;
  /** Degree of output graph. */
  size_t graph_degree = 64;
  /** ANN algorithm to build knn graph. */
  cagraGraphBuildAlgo build_algo = IVF_PQ;
  /** Number of Iterations to run if building with NN_DESCENT */
  size_t nn_descent_niter = 20;
};

typedef struct {
  uintptr_t addr;
  DLDataType data_type;

} cagraIndex;

void cagraIndexDestroy(cagraIndex index);

cagraIndex cagraBuild(raftResources_t res, cagraIndexParams params, DLManagedTensor* dataset);

#ifdef __cplusplus
}
#endif
