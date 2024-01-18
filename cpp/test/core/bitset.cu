/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <raft/core/bitset.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/linalg/init.cuh>
#include <raft/random/rng.cuh>

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>

namespace raft::core {

struct test_spec_bitset {
  uint64_t bitset_len;
  uint64_t mask_len;
  uint64_t query_len;
};

auto operator<<(std::ostream& os, const test_spec_bitset& ss) -> std::ostream&
{
  os << "bitset{bitset_len: " << ss.bitset_len << ", mask_len: " << ss.mask_len
     << ", query_len: " << ss.query_len << "}";
  return os;
}

template <typename bitset_t, typename index_t>
void add_cpu_bitset(std::vector<bitset_t>& bitset, const std::vector<index_t>& mask_idx)
{
  constexpr size_t bitset_element_size = sizeof(bitset_t) * 8;
  for (size_t i = 0; i < mask_idx.size(); i++) {
    auto idx = mask_idx[i];
    bitset[idx / bitset_element_size] &= ~(bitset_t{1} << (idx % bitset_element_size));
  }
}

template <typename bitset_t, typename index_t>
void create_cpu_bitset(std::vector<bitset_t>& bitset, const std::vector<index_t>& mask_idx)
{
  for (size_t i = 0; i < bitset.size(); i++) {
    bitset[i] = ~bitset_t(0x00);
  }
  add_cpu_bitset(bitset, mask_idx);
}

template <typename bitset_t, typename index_t>
void test_cpu_bitset(const std::vector<bitset_t>& bitset,
                     const std::vector<index_t>& queries,
                     std::vector<uint8_t>& result)
{
  constexpr size_t bitset_element_size = sizeof(bitset_t) * 8;
  for (size_t i = 0; i < queries.size(); i++) {
    result[i] = uint8_t((bitset[queries[i] / bitset_element_size] &
                         (bitset_t{1} << (queries[i] % bitset_element_size))) != 0);
  }
}

template <typename bitset_t>
void flip_cpu_bitset(std::vector<bitset_t>& bitset)
{
  for (size_t i = 0; i < bitset.size(); i++) {
    bitset[i] = ~bitset[i];
  }
}

template <typename bitset_t, typename index_t>
class BitsetTest : public testing::TestWithParam<test_spec_bitset> {
 protected:
  index_t static constexpr const bitset_element_size = sizeof(bitset_t) * 8;
  const test_spec_bitset spec;
  std::vector<bitset_t> bitset_result;
  std::vector<bitset_t> bitset_ref;
  raft::resources res;

 public:
  explicit BitsetTest()
    : spec(testing::TestWithParam<test_spec_bitset>::GetParam()),
      bitset_result(raft::ceildiv(spec.bitset_len, uint64_t(bitset_element_size))),
      bitset_ref(raft::ceildiv(spec.bitset_len, uint64_t(bitset_element_size)))
  {
  }

  void run()
  {
    auto stream = resource::get_cuda_stream(res);

    // generate input and mask
    raft::random::RngState rng(42);
    auto mask_device = raft::make_device_vector<index_t, index_t>(res, spec.mask_len);
    std::vector<index_t> mask_cpu(spec.mask_len);
    raft::random::uniformInt(res, rng, mask_device.view(), index_t(0), index_t(spec.bitset_len));
    update_host(mask_cpu.data(), mask_device.data_handle(), mask_device.extent(0), stream);
    resource::sync_stream(res, stream);

    // calculate the results
    auto my_bitset = raft::core::bitset<bitset_t, index_t>(
      res, raft::make_const_mdspan(mask_device.view()), index_t(spec.bitset_len));
    update_host(bitset_result.data(), my_bitset.data(), bitset_result.size(), stream);

    // calculate the reference
    create_cpu_bitset(bitset_ref, mask_cpu);
    resource::sync_stream(res, stream);
    ASSERT_TRUE(hostVecMatch(bitset_ref, bitset_result, raft::Compare<bitset_t>()));

    auto query_device  = raft::make_device_vector<index_t, index_t>(res, spec.query_len);
    auto result_device = raft::make_device_vector<uint8_t, index_t>(res, spec.query_len);
    auto query_cpu     = std::vector<index_t>(spec.query_len);
    auto result_cpu    = std::vector<uint8_t>(spec.query_len);
    auto result_ref    = std::vector<uint8_t>(spec.query_len);

    // Create queries and verify the test results
    raft::random::uniformInt(res, rng, query_device.view(), index_t(0), index_t(spec.bitset_len));
    update_host(query_cpu.data(), query_device.data_handle(), query_device.extent(0), stream);
    my_bitset.test(res, raft::make_const_mdspan(query_device.view()), result_device.view());
    update_host(result_cpu.data(), result_device.data_handle(), result_device.extent(0), stream);
    test_cpu_bitset(bitset_ref, query_cpu, result_ref);
    resource::sync_stream(res, stream);
    ASSERT_TRUE(hostVecMatch(result_cpu, result_ref, Compare<uint8_t>()));

    // Add more sample to the bitset and re-test
    raft::random::uniformInt(res, rng, mask_device.view(), index_t(0), index_t(spec.bitset_len));
    update_host(mask_cpu.data(), mask_device.data_handle(), mask_device.extent(0), stream);
    resource::sync_stream(res, stream);
    my_bitset.set(res, mask_device.view());
    update_host(bitset_result.data(), my_bitset.data(), bitset_result.size(), stream);

    add_cpu_bitset(bitset_ref, mask_cpu);
    resource::sync_stream(res, stream);
    ASSERT_TRUE(hostVecMatch(bitset_ref, bitset_result, raft::Compare<bitset_t>()));

    // Flip the bitset and re-test
    auto bitset_count = my_bitset.count(res);
    my_bitset.flip(res);
    ASSERT_EQ(my_bitset.count(res), spec.bitset_len - bitset_count);
    update_host(bitset_result.data(), my_bitset.data(), bitset_result.size(), stream);
    flip_cpu_bitset(bitset_ref);
    resource::sync_stream(res, stream);
    ASSERT_TRUE(hostVecMatch(bitset_ref, bitset_result, raft::Compare<bitset_t>()));

    // Test count() operations
    my_bitset.reset(res, false);
    ASSERT_EQ(my_bitset.any(res), false);
    ASSERT_EQ(my_bitset.none(res), true);
    raft::linalg::range(query_device.data_handle(), query_device.size(), stream);
    my_bitset.set(res, raft::make_const_mdspan(query_device.view()), true);
    bitset_count = my_bitset.count(res);
    ASSERT_EQ(bitset_count, query_device.size());
    ASSERT_EQ(my_bitset.any(res), true);
    ASSERT_EQ(my_bitset.none(res), false);
  }
};

auto inputs_bitset = ::testing::Values(test_spec_bitset{32, 5, 10},
                                       test_spec_bitset{100, 30, 10},
                                       test_spec_bitset{1024, 55, 100},
                                       test_spec_bitset{10000, 1000, 1000},
                                       test_spec_bitset{1 << 15, 1 << 3, 1 << 12},
                                       test_spec_bitset{1 << 15, 1 << 24, 1 << 13},
                                       test_spec_bitset{1 << 25, 1 << 23, 1 << 14});

using Uint16_32 = BitsetTest<uint16_t, uint32_t>;
TEST_P(Uint16_32, Run) { run(); }
INSTANTIATE_TEST_CASE_P(BitsetTest, Uint16_32, inputs_bitset);

using Uint32_32 = BitsetTest<uint32_t, uint32_t>;
TEST_P(Uint32_32, Run) { run(); }
INSTANTIATE_TEST_CASE_P(BitsetTest, Uint32_32, inputs_bitset);

using Uint64_32 = BitsetTest<uint64_t, uint32_t>;
TEST_P(Uint64_32, Run) { run(); }
INSTANTIATE_TEST_CASE_P(BitsetTest, Uint64_32, inputs_bitset);

using Uint8_64 = BitsetTest<uint8_t, uint64_t>;
TEST_P(Uint8_64, Run) { run(); }
INSTANTIATE_TEST_CASE_P(BitsetTest, Uint8_64, inputs_bitset);

using Uint32_64 = BitsetTest<uint32_t, uint64_t>;
TEST_P(Uint32_64, Run) { run(); }
INSTANTIATE_TEST_CASE_P(BitsetTest, Uint32_64, inputs_bitset);

using Uint64_64 = BitsetTest<uint64_t, uint64_t>;
TEST_P(Uint64_64, Run) { run(); }
INSTANTIATE_TEST_CASE_P(BitsetTest, Uint64_64, inputs_bitset);

}  // namespace raft::core
