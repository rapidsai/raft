#include "../test_utils.cuh"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <raft/random/rng.cuh>

namespace raft {
namespace random {

template<typename DType>
__host__ __device__ void buffer_fill() {
#ifdef __CUDA_ARCH__
  printf("Caling from device function %d\n", threadIdx.x);
#else
  printf("Calling from the host\n");
#endif
}

template<typename DType, typename ParamType, int TPB>
__global__ void demo_rng_kernel(RngState r, DType* buffer, size_t len, ParamType params) {

  int tid = int(blockIdx.x) * blockDim.x + threadIdx.x;
  int total_threads = int(blockDim.x) * gridDim.x;

  // current way of initializing PCG
  PCGenerator gen0(r.seed, r.base_subsequence + tid, tid);

  for (int i = 0; i < TPB; i++) {
    DType val;
    custom_next(gen0, &val, params, i, total_threads);
    if (tid*TPB + i < len) {
      buffer[tid*TPB + i] = val;
    }
  }
  if (tid == 123) buffer_fill<double>();

}

TEST(RNG, demo)
{
  cudaStream_t stream;

/*
  constexpr UniformDistParams<double> uniform_params = { .start = (0.0), .end = double(1.0)};

  constexpr UniformIntDistParams<int, uint32_t> uniform_int_params {.start = int(0), .end = int(1000), .diff = uint32_t(1000) };

constexpr NormalDistParams<double> normal_params = { .mu = double(0.5), .sigma = double(0.1)};

constexpr NormalIntDistParams<int> normal_params = {.mu = int(10), .sigma = int(5)};

constexpr NormalTableDistParams<int, double> . = { 
  LenType n_rows;
  LenType n_cols;
  const OutType* mu_vec;
  OutType sigma;
  const OutType* sigma_vec;
};

constexpr BernoulliDistParams<double> bernoulli_params = { .prob = double(0.75)};

constexpr ScaledBernoulliDistParams<double> scaled_bernoulli_params = {.prob = double(0.4), .scale = double(0.75) };

constexpr GumbelDistParams<double> _params = {
  .mu = double(),
  .beta = double()
};

constexpr LogNormalDistParams <double> log_normal_params = {
  .mu = double(),
  .sigma = double()
};

constexpr LogisticDistParams<double> logistic_params = {
  .mu = double(),
  .scale = double()
};

constexpr ExponentialDistParams<double> exponential_params = {
  .lambda = double()
};

constexpr RayleighDistParams<double> rayleigh_params = {
  .sigma = double()
};

constexpr LaplaceDistParams<double> laplace_params = {
  .mu = double(),
  .scale = double()
};
*/

  raft::resources handle;
  stream = resource::get_cuda_stream(handle);

  using dtype          = double;
  constexpr size_t len = size_t(128) * 1024 * 1024;

  rmm::device_uvector<dtype> d_buffer(len, stream);
  std::vector<dtype> h_buffer(len);
  std::vector<dtype> buffer(len);

  
  RngState r(uint64_t(0x4d5cd6fc9a66e8c2), GenPC);

  int n_blocks = 128;
  int n_threads = 64;
  constexpr int TPB = 16;
  int total_threads = n_blocks * n_threads;
  constexpr InvariantDistParams<double> invariant_params = {.const_val = double(42.0)};
  demo_rng_kernel<double, InvariantDistParams<double>, TPB><<<n_blocks, n_threads>>>(r, d_buffer.data(), len, invariant_params);

  update_host(h_buffer.data(), d_buffer.data(), len, stream);

  for(int tid = 0; tid < total_threads; tid++) {
    PCGenerator gen0(r.seed, r.base_subsequence + tid, tid);
    dtype val = 0.0; 
    for (int i = 0; i < TPB; i++) {
      custom_next(gen0, &val, invariant_params, i, 1);
      if (tid*TPB + i < int(len)) {
        buffer[tid*TPB + i] = val;
      }
    }
  }
  /*for (int i = 0; i < 2*TPB; i++) {
    printf("%f - %f\n", buffer[i], h_buffer[i]);
  }*/
  
  ASSERT_TRUE(devArrMatchHost(buffer.data(), d_buffer.data(), total_threads, raft::Compare<double>(), stream));
  buffer_fill<double>();
  

}

} // namespace random
} // namespace raft
