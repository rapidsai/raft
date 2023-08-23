#include "../test_utils.cuh"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <raft/random/rng.cuh>

namespace raft {
namespace random {

// CPT - Calls Per Thread, How many calls to custom_next is made by a single thread
// IPC - Items Per Call, How many items are returned by a single call to custom_next (usually IPC =
// 1 or 2)
template <typename DType, typename ParamType, int CPT, int IPC>
__host__ __device__ void single_thread_fill(DType* buffer,
                                            DeviceState<PCGenerator> r,
                                            ParamType params,
                                            const size_t total_threads,
                                            const size_t len,
                                            const size_t tid)
{
  PCGenerator gen(r, tid);

  for (size_t i = 0; i < CPT; i++) {
    DType val[IPC];
    size_t index = (tid * CPT * IPC) + i * IPC;
    custom_next(gen, val, params, index, total_threads);
    for (int j = 0; j < IPC; j++) {
      if (index + j < len) { buffer[index + j] = val[j]; }
    }
  }
}

template <typename DType, typename ParamType, int CPT, int IPC>
__global__ void pcg_device_kernel(DType* buffer,
                                  DeviceState<PCGenerator> r,
                                  ParamType params,
                                  const size_t total_threads,
                                  const size_t len)
{
  int tid = int(blockIdx.x) * blockDim.x + threadIdx.x;

  single_thread_fill<DType, ParamType, CPT, IPC>(buffer, r, params, total_threads, len, tid);
}

template <typename ParamType, typename DataType, int CPT, int IPC>
class HostApiTest {
 public:
  HostApiTest() : stream(resource::get_cuda_stream(handle)), d_buffer(0, stream)
  {
    len = total_threads * CPT * IPC;
    d_buffer.resize(len, stream);
    h_buffer.resize(len);
    printf("### len = %lu\n", len);
  }
  void FillBuffers(uint64_t seed)
  {
    printf("seed = %lu\n", seed);
    RngState r(seed, GenPC);
    DeviceState<PCGenerator> d_state(r);

    pcg_device_kernel<DataType, ParamType, CPT, IPC><<<n_blocks, n_threads, 0, stream>>>(
      d_buffer.data(), d_state, dist_params, total_threads, len);

    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    for (size_t tid = 0; tid < len; tid++) {
      single_thread_fill<DataType, ParamType, CPT, IPC>(
        h_buffer.data(), d_state, dist_params, total_threads, len, tid);
    }
  }
  void SetParams(ParamType _dist_params) { dist_params = _dist_params; }

  void test()
  {
    ASSERT_TRUE(devArrMatchHost(
      h_buffer.data(), d_buffer.data(), len, raft::CompareApprox<double>(1e-5), stream));
  }
  ParamType dist_params;
  raft::resources handle;
  cudaStream_t stream;

  static const int n_blocks         = 128;
  static const int n_threads        = 64;
  static const size_t total_threads = size_t(n_blocks) * n_threads;

  size_t len;
  rmm::device_uvector<DataType> d_buffer;
  std::vector<DataType> h_buffer;
};

template <typename T>
class TestWrapper : public testing::Test {
 protected:
  void SetUp() override
  {
    test_obj.SetParams(p);
    test_obj.FillBuffers(seed);
  }

 public:
  void print_foo() { test_obj.test(); }
  T test_obj;
  using ParamType = decltype(T::dist_params);
  static ParamType p;
  const uint64_t seed = 42;
};

TYPED_TEST_SUITE_P(TestWrapper);

TYPED_TEST_P(TestWrapper, print) { this->print_foo(); }

REGISTER_TYPED_TEST_SUITE_P(TestWrapper, print);

using InvariantDistType = HostApiTest<InvariantDistParams<int>, int, 16, 1>;
template <>
InvariantDistParams<int> TestWrapper<InvariantDistType>::p = {.const_val = 431601};

using UniformDistType = HostApiTest<UniformDistParams<double>, double, 16, 1>;
template <>
UniformDistParams<double> TestWrapper<UniformDistType>::p = {.start = 0.0, .end = 1.0};

using UniformInt32DistType = HostApiTest<UniformIntDistParams<uint32_t, uint32_t>, uint32_t, 16, 1>;
template <>
UniformIntDistParams<uint32_t, uint32_t> TestWrapper<UniformInt32DistType>::p = {
  .start = 0, .end = 100000, .diff = 100000};

using UniformInt64DistType = HostApiTest<UniformIntDistParams<uint64_t, uint64_t>, uint64_t, 16, 1>;
template <>
UniformIntDistParams<uint64_t, uint64_t> TestWrapper<UniformInt64DistType>::p = {
  .start = 0, .end = 100000, .diff = 100000};

using NormalDistType = HostApiTest<NormalDistParams<double>, double, 16, 2>;
template <>
NormalDistParams<double> TestWrapper<NormalDistType>::p = {.mu = 0.5, .sigma = 0.5};

using NormalIntDistType = HostApiTest<NormalIntDistParams<uint32_t>, uint32_t, 16, 2>;
template <>
NormalIntDistParams<uint32_t> TestWrapper<NormalIntDistType>::p = {.mu = 1, .sigma = 1};

using BernoulliDistType = HostApiTest<BernoulliDistParams<double>, double, 16, 1>;
template <>
BernoulliDistParams<double> TestWrapper<BernoulliDistType>::p = {.prob = 0.7};

using ScaledBernoulliDistType = HostApiTest<ScaledBernoulliDistParams<double>, double, 16, 1>;
template <>
ScaledBernoulliDistParams<double> TestWrapper<ScaledBernoulliDistType>::p = {.prob  = 0.7,
                                                                             .scale = 0.5};

using GumbelDistType = HostApiTest<GumbelDistParams<double>, double, 16, 1>;
template <>
GumbelDistParams<double> TestWrapper<GumbelDistType>::p = {.mu = 0.7, .beta = 0.5};

using LogNormalDistType = HostApiTest<LogNormalDistParams<double>, double, 16, 2>;
template <>
LogNormalDistParams<double> TestWrapper<LogNormalDistType>::p = {.mu = 0.5, .sigma = 0.5};

using LogisticDistType = HostApiTest<LogisticDistParams<double>, double, 16, 1>;
template <>
LogisticDistParams<double> TestWrapper<LogisticDistType>::p = {.mu = 0.2, .scale = 0.3};

using ExponentialDistType = HostApiTest<ExponentialDistParams<double>, double, 16, 1>;
template <>
ExponentialDistParams<double> TestWrapper<ExponentialDistType>::p = {.lambda = 1.6};

using RayleighDistType = HostApiTest<RayleighDistParams<double>, double, 16, 1>;
template <>
RayleighDistParams<double> TestWrapper<RayleighDistType>::p = {.sigma = 1.6};

using LaplaceDistType = HostApiTest<LaplaceDistParams<double>, double, 16, 1>;
template <>
LaplaceDistParams<double> TestWrapper<LaplaceDistType>::p = {.mu = 0.2, .scale = 0.3};

using TestingTypes1 = testing::Types<InvariantDistType,
                                     UniformDistType,
                                     UniformInt32DistType,
                                     /*UniformInt64DistType, */ NormalDistType,
                                     /*NormalIntDistType, */ BernoulliDistType,
                                     ScaledBernoulliDistType,
                                     GumbelDistType,
                                     LogisticDistType,
                                     ExponentialDistType,
                                     RayleighDistType,
                                     LaplaceDistType>;

INSTANTIATE_TYPED_TEST_SUITE_P(My1, TestWrapper, TestingTypes1);

}  // namespace random
}  // namespace raft
