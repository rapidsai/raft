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
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>

#include <raft/core/host_mdarray.hpp>
#include <raft/core/interop.hpp>

#include <gtest/gtest.h>
#include <sys/types.h>

namespace raft::core {

TEST(Interop, FromDLPack)
{
  raft::resources res;
  auto data = raft::make_host_vector<float>(res, 2);
  data(0)   = 5;
  data(1)   = 10;

  auto device    = DLDevice{kDLCPU};
  auto data_type = DLDataType{kDLFloat, 4 * 8, 1};
  auto shape     = std::vector<int64_t>{2};

  auto tensor         = DLTensor{data.data_handle(), device, 1, data_type, shape.data()};
  auto managed_tensor = DLManagedTensor{tensor};

  using mdspan_type = raft::host_mdspan<float const, raft::vector_extent<int64_t>>;
  auto out          = from_dlpack<mdspan_type>(&managed_tensor);

  ASSERT_EQ(out.rank(), data.rank());
  ASSERT_EQ(out.extent(0), data.extent(0));
  ASSERT_EQ(out(0), data(0));
  ASSERT_EQ(out(1), data(1));
}

}  // namespace raft::core
