/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#include <gtest/gtest.h>
#include <raft/interruptible.hpp>
#include <raft/linalg/reduce_cols_by_key.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

namespace raft {
namespace linalg {

template <typename T>
void naiveReduceColsByKey(const T* in,
                          const uint32_t* keys,
                          T* out_ref,
                          uint32_t nrows,
                          uint32_t ncols,
                          uint32_t nkeys,
                          cudaStream_t stream)
{
  std::vector<uint32_t> h_keys(ncols, 0u);
  raft::copy(&(h_keys[0]), keys, ncols, stream);
  std::vector<T> h_in(nrows * ncols);
  raft::copy(&(h_in[0]), in, nrows * ncols, stream);
  raft::interruptible::synchronize(stream);
  std::vector<T> out(nrows * nkeys, T(0));
  for (uint32_t i = 0; i < nrows; ++i) {
    for (uint32_t j = 0; j < ncols; ++j) {
      out[i * nkeys + h_keys[j]] += h_in[i * ncols + j];
    }
  }
  raft::copy(out_ref, &(out[0]), nrows * nkeys, stream);
  raft::interruptible::synchronize(stream);
}

template <typename T>
struct ReduceColsInputs {
  T tolerance;
  uint32_t rows;
  uint32_t cols;
  uint32_t nkeys;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const ReduceColsInputs<T>& dims)
{
  return os;
}

template <typename T>
class ReduceColsTest : public ::testing::TestWithParam<ReduceColsInputs<T>> {
 protected:
  ReduceColsTest() : in(0, stream), out_ref(0, stream), out(0, stream), keys(0, stream) {}

  void SetUp() override
  {
    params = ::testing::TestWithParam<ReduceColsInputs<T>>::GetParam();
    raft::random::RngState r(params.seed);
    raft::handle_t handle;
    auto stream = handle.get_stream();
    auto nrows  = params.rows;
    auto ncols  = params.cols;
    auto nkeys  = params.nkeys;
    in.resize(nrows * ncols, stream);
    keys.resize(ncols, stream);
    out_ref.resize(nrows * nkeys, stream);
    out.resize(nrows * nkeys, stream);
    uniform(handle, r, in.data(), nrows * ncols, T(-1.0), T(1.0));
    uniformInt(handle, r, keys.data(), ncols, 0u, params.nkeys);
    naiveReduceColsByKey(in.data(), keys.data(), out_ref.data(), nrows, ncols, nkeys, stream);
    auto input_view  = raft::make_device_matrix_view<const T>(in.data(), nrows, ncols);
    auto output_view = raft::make_device_matrix_view(out.data(), nrows, nkeys);
    auto keys_view   = raft::make_device_vector_view<const uint32_t>(keys.data(), ncols);
    reduce_cols_by_key(handle, input_view, keys_view, output_view, nkeys);
    raft::interruptible::synchronize(stream);
  }

 protected:
  cudaStream_t stream = 0;
  ReduceColsInputs<T> params;
  rmm::device_uvector<T> in, out_ref, out;
  rmm::device_uvector<uint32_t> keys;
};

const std::vector<ReduceColsInputs<float>> inputsf = {{0.0001f, 128, 32, 6, 1234ULL},
                                                      {0.0005f, 121, 63, 10, 1234ULL}};
typedef ReduceColsTest<float> ReduceColsTestF;
TEST_P(ReduceColsTestF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(out_ref.data(),
                                out.data(),
                                params.rows * params.nkeys,
                                raft::CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(ReduceColsTests, ReduceColsTestF, ::testing::ValuesIn(inputsf));

const std::vector<ReduceColsInputs<double>> inputsd2 = {{0.0000001, 128, 32, 6, 1234ULL},
                                                        {0.0000001, 121, 63, 10, 1234ULL}};
typedef ReduceColsTest<double> ReduceColsTestD;
TEST_P(ReduceColsTestD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(out_ref.data(),
                                out.data(),
                                params.rows * params.nkeys,
                                raft::CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(ReduceColsTests, ReduceColsTestD, ::testing::ValuesIn(inputsd2));

}  // end namespace linalg
}  // end namespace raft
