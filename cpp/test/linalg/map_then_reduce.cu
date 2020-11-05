/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

#include <gtest/gtest.h>
#include <raft/cudart_utils.h>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/random/rng.cuh>
#include "../test_utils.h"

namespace raft {
namespace linalg {

template <typename Type, typename MapOp>
__global__ void naive_map_reduce_kernel(Type *out, const Type *in, size_t len,
                                     MapOp map) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) {
    raft::myAtomicAdd(out, map(in[idx]));
  }
}

template <typename Type, typename MapOp>
void naive_map_reduce(Type *out, const Type *in, size_t len, MapOp map,
                    cudaStream_t stream) {
  static const int kTpb = 64;
  int nblks = raft::ceildiv(len, static_cast<size_t>(kTpb));
  naive_map_reduce_kernel<Type, MapOp>
    <<<nblks, kTpb, 0, stream>>>(out, in, len, map);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
struct map_reduce_inputs {
  T tolerance;
  size_t len;
  uint64_t seed;
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os, const map_reduce_inputs<T> &dims) {
  return os;
}

// Or else, we get the following compilation error
// for an extended __device__ lambda cannot have private or protected access
// within its class
template <typename T>
void map_reduce_launch(T *out_ref, T *out, const T *in, size_t len,
                     cudaStream_t stream) {
  auto op = [] __device__(T in) { return in; };
  naive_map_reduce(out_ref, in, len, op, stream);
  mapThenSumReduce(out, len, op, 0, in);
}

template <typename T>
class map_reduce_test : public ::testing::TestWithParam<map_reduce_inputs<T>> {
 protected:
  void SetUp() override {  // NOLINT
    params_ = ::testing::TestWithParam<map_reduce_inputs<T>>::GetParam();
    raft::random::Rng r(params_.seed);
    auto len = params_.len;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    allocate(in_, len);
    allocate(out_ref_, len);
    allocate(out_, len);
    r.uniform(in_, len, T(-1.0), T(1.0), stream);
    map_reduce_launch(out_ref_, out_, in_, len, stream);
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void TearDown() override {  // NOLINT
    CUDA_CHECK(cudaFree(in_));
    CUDA_CHECK(cudaFree(out_ref_));
    CUDA_CHECK(cudaFree(out_));
  }

  map_reduce_inputs<T> params_;
  T *in_, *out_ref_, *out_;
};

const std::vector<map_reduce_inputs<float>> kInputsF = {
  {0.001f, 1024 * 1024, 1234ULL}};
using map_reduce_test_f = map_reduce_test<float>;
TEST_P(map_reduce_test_f, Result) {  // NOLINT
  ASSERT_TRUE(devArrMatch(out_ref_, out_, params_.len,
                          compare_approx<float>(params_.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(map_reduce_tests, map_reduce_test_f,  // NOLINT
                         ::testing::ValuesIn(kInputsF));

const std::vector<map_reduce_inputs<double>> kInputsD = {
  {0.000001, 1024 * 1024, 1234ULL}};
using map_reduce_test_d = map_reduce_test<double>;
TEST_P(map_reduce_test_d, Result) {  // NOLINT
  ASSERT_TRUE(devArrMatch(out_ref_, out_, params_.len,
                          compare_approx<double>(params_.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(map_reduce_tests, map_reduce_test_d,  // NOLINT
                         ::testing::ValuesIn(kInputsD));

}  // end namespace linalg
}  // end namespace raft
