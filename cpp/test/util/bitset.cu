/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <raft/random/rng.cuh>
#include <raft/util/bitset.cuh>

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>

namespace raft::utils {

struct test_spec {
  int bitset_len;
  int mask_len;
  int query_len;
};

auto operator<<(std::ostream& os, const test_spec& ss) -> std::ostream&
{
  os << "bitset{bitset_len: " << ss.bitset_len << ", mask_len: " << ss.mask_len << "}";
  return os;
}

template <typename T>
void create_cpu_bitset(std::vector<uint32_t>& bitset, const std::vector<T>& mask_idx)
{
  for (size_t i = 0; i < bitset.size(); i++) {
    bitset[i] = 0xffffffff;
  }
  for (size_t i = 0; i < mask_idx.size(); i++) {
    auto idx = mask_idx[i];
    bitset[idx / 32] &= ~(1 << (idx % 32));
  }
}

template <typename T>
void test_cpu_bitset(const std::vector<uint32_t>& bitset,
                     const std::vector<T>& queries,
                     std::vector<uint8_t>& result)
{
  for (size_t i = 0; i < queries.size(); i++) {
    result[i] = uint8_t((bitset[queries[i] / 32] & (1 << (queries[i] % 32))) != 0);
  }
}

template <typename IdxT>
__global__ void test_gpu_bitset(bitset_view<IdxT> bitset,
                                const IdxT* queries,
                                uint8_t* result,
                                IdxT n_queries)
{
  for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < n_queries;
       tid += blockDim.x * gridDim.x) {
    auto query  = queries[tid];
    result[tid] = (uint8_t)bitset.test(query);
  }
}

template <typename T>
class BitsetTest : public testing::TestWithParam<test_spec> {
 protected:
  const test_spec spec;
  std::vector<uint32_t> bitset_result;
  std::vector<uint32_t> bitset_ref;
  raft::resources res;

 public:
  explicit BitsetTest()
    : spec(testing::TestWithParam<test_spec>::GetParam()),
      bitset_result(raft::ceildiv(spec.bitset_len, 32)),
      bitset_ref(raft::ceildiv(spec.bitset_len, 32))
  {
  }

  void run()
  {
    auto stream = resource::get_cuda_stream(res);

    // generate input and mask
    raft::random::RngState rng(42);
    auto mask_device = raft::make_device_vector<T, T>(res, spec.mask_len);
    std::vector<T> mask_cpu(spec.mask_len);
    raft::random::uniformInt(res, rng, mask_device.view(), T(0), T(spec.bitset_len));
    update_host(mask_cpu.data(), mask_device.data_handle(), mask_device.extent(0), stream);
    resource::sync_stream(res, stream);

    // calculate the results
    auto test_bitset =
      raft::utils::bitset<T>(res, raft::make_const_mdspan(mask_device.view()), T(spec.bitset_len));
    update_host(
      bitset_result.data(), test_bitset.view().get_bitset_ptr(), bitset_result.size(), stream);

    // calculate the reference
    create_cpu_bitset(bitset_ref, mask_cpu);

    // make sure the results are available on host
    resource::sync_stream(res, stream);
    ASSERT_TRUE(hostVecMatch(bitset_ref, bitset_result, raft::Compare<T>()));

    auto query_device  = raft::make_device_vector<T, T>(res, spec.query_len);
    auto result_device = raft::make_device_vector<uint8_t, T>(res, spec.query_len);
    auto query_cpu     = std::vector<T>(spec.query_len);
    auto result_cpu    = std::vector<uint8_t>(spec.query_len);
    auto result_ref    = std::vector<uint8_t>(spec.query_len);

    raft::random::uniformInt(res, rng, query_device.view(), T(0), T(spec.bitset_len));
    update_host(query_cpu.data(), query_device.data_handle(), query_device.extent(0), stream);
    test_gpu_bitset<<<spec.query_len, 128, 0, stream>>>(test_bitset.view(),
                                                        query_device.data_handle(),
                                                        result_device.data_handle(),
                                                        query_device.extent(0));
    update_host(result_cpu.data(), result_device.data_handle(), result_device.extent(0), stream);
    test_cpu_bitset(bitset_ref, query_cpu, result_ref);

    resource::sync_stream(res, stream);
    ASSERT_TRUE(hostVecMatch(result_cpu, result_ref, Compare<uint8_t>()));
  }
};

auto inputs = ::testing::Values(test_spec{1 << 25, 1 << 23, 1 << 24},
                                test_spec{32, 5, 10},
                                test_spec{100, 30, 10},
                                test_spec{1024, 55, 100},
                                test_spec{10000, 1000, 1000},
                                test_spec{1 << 15, 1 << 3, 1 << 12},
                                test_spec{1 << 15, 1 << 14, 1 << 13});

using Uint32 = BitsetTest<uint32_t>;
TEST_P(Uint32, Run) { run(); }
INSTANTIATE_TEST_CASE_P(BitsetTest, Uint32, inputs);

using Uint64 = BitsetTest<uint64_t>;
TEST_P(Uint64, Run) { run(); }
INSTANTIATE_TEST_CASE_P(BitsetTest, Uint64, inputs);

}  // namespace raft::utils
