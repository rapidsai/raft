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

#include "../test_utils.cuh"
#include <dlpack/dlpack.h>
#include <raft/core/c_api.h>

#include <cstdint>
#include <raft/neighbors/cagra.h>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

float dataset[4][2] = {{0.74021935, 0.9209938},
                       {0.03902049, 0.9689629},
                       {0.92514056, 0.4463501},
                       {0.6673192, 0.10993068}};
float queries[4][2] = {{0.48216683, 0.0428398},
                       {0.5084142, 0.6545497},
                       {0.51260436, 0.2643005},
                       {0.05198065, 0.5789965}};

uint32_t neighbors_exp[4] = {3, 0, 3, 1};
float distances_exp[4]    = {0.03878258, 0.12472608, 0.04776672, 0.15224178};

TEST(CagraC, BuildSearch)
{
  // create raftResources_t
  raftResources_t res;
  raftCreateResources(&res);

  // create dataset DLTensor
  DLManagedTensor dataset_tensor;
  dataset_tensor.dl_tensor.data               = dataset;
  dataset_tensor.dl_tensor.device.device_type = kDLCPU;
  dataset_tensor.dl_tensor.ndim               = 2;
  dataset_tensor.dl_tensor.dtype.code         = kDLFloat;
  dataset_tensor.dl_tensor.dtype.bits         = 32;
  dataset_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t dataset_shape[2]                    = {4, 2};
  dataset_tensor.dl_tensor.shape              = dataset_shape;
  dataset_tensor.dl_tensor.strides            = nullptr;

  // build index
  cagraIndexParams build_params;
  cagraIndex index = cagraBuild(res, build_params, &dataset_tensor);

  // create queries DLTensor
  float* queries_d;
  cudaMalloc(&queries_d, sizeof(float) * 4 * 2);
  cudaMemcpy(queries_d, queries, sizeof(float) * 4 * 2, cudaMemcpyDefault);

  DLManagedTensor queries_tensor;
  queries_tensor.dl_tensor.data               = queries_d;
  queries_tensor.dl_tensor.device.device_type = kDLCUDA;
  queries_tensor.dl_tensor.ndim               = 2;
  queries_tensor.dl_tensor.dtype.code         = kDLFloat;
  queries_tensor.dl_tensor.dtype.bits         = 32;
  queries_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t queries_shape[2]                    = {4, 2};
  queries_tensor.dl_tensor.shape              = queries_shape;
  queries_tensor.dl_tensor.strides            = nullptr;

  // create neighbors DLTensor
  uint32_t* neighbors_d;
  cudaMalloc(&neighbors_d, sizeof(uint32_t) * 4);

  DLManagedTensor neighbors_tensor;
  neighbors_tensor.dl_tensor.data               = neighbors_d;
  neighbors_tensor.dl_tensor.device.device_type = kDLCUDA;
  neighbors_tensor.dl_tensor.ndim               = 2;
  neighbors_tensor.dl_tensor.dtype.code         = kDLUInt;
  neighbors_tensor.dl_tensor.dtype.bits         = 32;
  neighbors_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t neighbors_shape[2]                    = {4, 1};
  neighbors_tensor.dl_tensor.shape              = neighbors_shape;
  neighbors_tensor.dl_tensor.strides            = nullptr;

  // create distances DLTensor
  float* distances_d;
  cudaMalloc(&distances_d, sizeof(float) * 4);

  DLManagedTensor distances_tensor;
  distances_tensor.dl_tensor.data               = distances_d;
  distances_tensor.dl_tensor.device.device_type = kDLCUDA;
  distances_tensor.dl_tensor.ndim               = 2;
  distances_tensor.dl_tensor.dtype.code         = kDLFloat;
  distances_tensor.dl_tensor.dtype.bits         = 32;
  distances_tensor.dl_tensor.dtype.lanes        = 1;
  int64_t distances_shape[2]                    = {4, 1};
  distances_tensor.dl_tensor.shape              = distances_shape;
  distances_tensor.dl_tensor.strides            = nullptr;

  // search index
  cagraSearchParams search_params;
  cagraSearch(res, search_params, index, &queries_tensor, &neighbors_tensor, &distances_tensor);

  // verify output
  ASSERT_TRUE(raft::devArrMatchHost(neighbors_exp, neighbors_d, 4, raft::Compare<uint32_t>()));
  ASSERT_TRUE(
    raft::devArrMatchHost(distances_exp, distances_d, 4, raft::CompareApprox<float>(0.001f)));

  // delete device memory
  cudaFree(queries_d);
  cudaFree(neighbors_d);
  cudaFree(distances_d);

  // de-allocate index and res
  cagraDestroyIndex(index);
  raftDestroyResources(res);
}
