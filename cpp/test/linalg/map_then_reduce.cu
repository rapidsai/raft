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
#include <limits>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/random/rng.cuh>
#include "../test_utils.h"

namespace raft {
namespace linalg {

template <typename Type, typename MapOp>
__global__ void naiveMapReduceKernel(Type *out, const Type *in, size_t len,
                                     MapOp map) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) {
    raft::myAtomicAdd(out, map(in[idx]));
  }
}

template <typename Type, typename MapOp>
void naiveMapReduce(Type *out, const Type *in, size_t len, MapOp map,
                    cudaStream_t stream) {
  static const int TPB = 64;
  int nblks = raft::ceildiv(len, (size_t)TPB);
  naiveMapReduceKernel<Type, MapOp>
    <<<nblks, TPB, 0, stream>>>(out, in, len, map);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
struct MapReduceInputs {
  T tolerance;
  size_t len;
  unsigned long long int seed;
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os, const MapReduceInputs<T> &dims) {
  return os;
}

// Or else, we get the following compilation error
// for an extended __device__ lambda cannot have private or protected access
// within its class
template <typename T>
void mapReduceLaunch(T *out_ref, T *out, const T *in, size_t len,
                     cudaStream_t stream) {
  auto op = [] __device__(T in) { return in; };
  naiveMapReduce(out_ref, in, len, op, stream);
  mapThenSumReduce(out, len, op, 0, in);
}

template <typename T>
class MapReduceTest : public ::testing::TestWithParam<MapReduceInputs<T>> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<MapReduceInputs<T>>::GetParam();
    raft::random::Rng r(params.seed);
    auto len = params.len;

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    allocate(in, len);
    allocate(out_ref, len);
    allocate(out, len);
    r.uniform(in, len, T(-1.0), T(1.0), stream);
    mapReduceLaunch(out_ref, out, in, len, stream);
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(in));
    CUDA_CHECK(cudaFree(out_ref));
    CUDA_CHECK(cudaFree(out));
  }

 protected:
  MapReduceInputs<T> params;
  T *in, *out_ref, *out;
};

const std::vector<MapReduceInputs<float>> inputsf = {
  {0.001f, 1024 * 1024, 1234ULL}};
typedef MapReduceTest<float> MapReduceTestF;
TEST_P(MapReduceTestF, Result) {
  ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                          CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(MapReduceTests, MapReduceTestF,
                         ::testing::ValuesIn(inputsf));

const std::vector<MapReduceInputs<double>> inputsd = {
  {0.000001, 1024 * 1024, 1234ULL}};
typedef MapReduceTest<double> MapReduceTestD;
TEST_P(MapReduceTestD, Result) {
  ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                          CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(MapReduceTests, MapReduceTestD,
                         ::testing::ValuesIn(inputsd));

template <typename math_t>
class MapGenericReduceTest : public ::testing::Test {
 protected:
  MapGenericReduceTest()
    : allocator(handle.get_device_allocator()),
      input(allocator, handle.get_stream(), n),
      output(allocator, handle.get_stream(), 1) {
    CUDA_CHECK(cudaStreamCreate(&stream));
    handle.set_stream(stream);
    initInput(input.data(), input.size(), stream);
  }

  void TearDown() override { CUDA_CHECK(cudaStreamDestroy(stream)); }

 public:
  void initInput(math_t *input, int n, cudaStream_t stream) {
    raft::random::Rng r(137);
    r.uniform(input, n, math_t(2), math_t(3), stream);
    math_t val = 1;
    raft::update_device(input + 42, &val, 1, stream);
    val = 5;
    raft::update_device(input + 337, &val, 1, stream);
  }

  void testMin() {
    auto op = [] __device__(math_t in) { return in; };
    const math_t neutral = std::numeric_limits<math_t>::max();
    mapThenReduce(output.data(), input.size(), neutral, op, cub::Min(), stream,
                  input.data());
    EXPECT_TRUE(
      raft::devArrMatch(math_t(1), output.data(), 1, raft::Compare<math_t>()));
  }
  void testMax() {
    auto op = [] __device__(math_t in) { return in; };
    const math_t neutral = std::numeric_limits<math_t>::min();
    mapThenReduce(output.data(), input.size(), neutral, op, cub::Max(), stream,
                  input.data());
    EXPECT_TRUE(
      raft::devArrMatch(math_t(5), output.data(), 1, raft::Compare<math_t>()));
  }

 protected:
  int n = 1237;
  raft::handle_t handle;
  cudaStream_t stream;
  std::shared_ptr<raft::mr::device::allocator> allocator;
  raft::mr::device::buffer<math_t> input;
  raft::mr::device::buffer<math_t> output;
};

typedef ::testing::Types<float, double> FloatTypes;

TYPED_TEST_CASE(MapGenericReduceTest, FloatTypes);
TYPED_TEST(MapGenericReduceTest, min) { this->testMin(); }
TYPED_TEST(MapGenericReduceTest, max) { this->testMax(); }
}  // end namespace linalg
}  // end namespace raft
