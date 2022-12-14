/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <faiss/gpu/GpuResources.h>
#include <raft/distance/distance_types.hpp>
#include <raft/spatial/knn/knn.cuh>

#include <rmm/device_buffer.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <iostream>
#include <vector>

namespace raft {
namespace spatial {
namespace knn {

using namespace faiss::gpu;

struct AllocInputs {
  size_t size;
};

template <typename T>
class FAISS_MR_Test : public ::testing::TestWithParam<AllocInputs> {
 public:
  FAISS_MR_Test()
    : params_(::testing::TestWithParam<AllocInputs>::GetParam()), stream(handle.get_stream())
  {
  }

 protected:
  size_t getFreeMemory(MemorySpace mem_space)
  {
    if (mem_space == MemorySpace::Device) {
      rmm::mr::cuda_memory_resource cmr;
      rmm::mr::device_memory_resource* dmr = &cmr;
      return dmr->get_mem_info(stream).first;
    } else if (mem_space == MemorySpace::Unified) {
      rmm::mr::managed_memory_resource mmr;
      rmm::mr::device_memory_resource* dmr = &mmr;
      return dmr->get_mem_info(stream).first;
    }
    return 0;
  }

  void testAllocs(MemorySpace mem_space)
  {
    raft::spatial::knn::RmmGpuResources faiss_mr;
    auto faiss_mr_impl = faiss_mr.getResources();
    size_t free_before = getFreeMemory(mem_space);
    AllocRequest req(AllocType::Other, 0, mem_space, stream, params_.size);
    void* ptr               = faiss_mr_impl->allocMemory(req);
    size_t free_after_alloc = getFreeMemory(mem_space);
    faiss_mr_impl->deallocMemory(0, ptr);
    ASSERT_TRUE(free_after_alloc <= free_before - params_.size);
  }

  raft::handle_t handle;
  cudaStream_t stream;
  AllocInputs params_;
};

const std::vector<AllocInputs> inputs = {{19687}};

typedef FAISS_MR_Test<float> FAISS_MR_TestF;
TEST_P(FAISS_MR_TestF, TestAllocs)
{
  testAllocs(MemorySpace::Device);
  testAllocs(MemorySpace::Unified);
}

INSTANTIATE_TEST_CASE_P(FAISS_MR_Test, FAISS_MR_TestF, ::testing::ValuesIn(inputs));

}  // namespace knn
}  // namespace spatial
}  // namespace raft
