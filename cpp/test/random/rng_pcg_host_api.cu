#include "../test_utils.cuh"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <raft/random/rng.cuh>

namespace raft {
namespace random {

// CPT - Calls Per Thread, How many calls to custom_next is made by a single thread
// IPC - Items Per Call, How many items are returned by a single call to custom_next (usually IPC = 1 or 2)
template<typename DType, typename ParamType, int CPT, int IPC>
__host__ __device__ void single_thread_fill(DType* buffer, DeviceState<PCGenerator> r, ParamType params, const size_t total_threads, const size_t len, const size_t tid) {

  PCGenerator gen(r, tid);
 
  for (size_t i = 0; i < CPT; i++) {
    DType val[IPC];
    size_t index = (tid * CPT * IPC) + i * IPC;
    custom_next(gen, val, params, index, total_threads);
    for (int j = 0; j < IPC; j++) {
      if (index + j < len) {
        buffer[index + j] = val[j];
      }
    }
  }
}

template<typename DType, typename ParamType, int CPT, int IPC>
__global__ void pcg_device_kernel(DType* buffer, DeviceState<PCGenerator> r, ParamType params, const size_t total_threads, const size_t len) {
  int tid = int(blockIdx.x) * blockDim.x + threadIdx.x;

  single_thread_fill<DType, ParamType, CPT, IPC>(buffer, r, params, total_threads, len, tid);

}

/*void trying_func(){
  constexpr int IPC = 1;
  constexpr size_t len = total_threads * CPT * IPC;
  printf("len = %lu\n", len);
  UniformDistParams<T> uniform_params = { .start = params.start, .end = params.end};

  d_buffer.resize(len, stream);
  h_buffer.resize(len);

  pcg_device_kernel<T, UniformDistParams<T>, CPT, IPC><<<n_blocks, n_threads>>>(d_buffer.data(), d_state, uniform_params, total_threads, len);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  for(size_t tid = 0; tid < total_threads; tid++) {
    single_thread_fill<T, UniformDistParams<T>, CPT, IPC>(h_buffer.data(), d_state, uniform_params, total_threads, len, tid);
  }
}*/

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
  //demo_rng_kernel<double, InvariantDistParams<double>, TPB><<<n_blocks, n_threads>>>(r, d_buffer.data(), len, invariant_params);

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

}

enum RandomType {
  RNG_Normal,
  RNG_LogNormal,
  RNG_Uniform,
  RNG_Gumbel,
  RNG_Logistic,
  RNG_Exp,
  RNG_Rayleigh,
  RNG_Laplace
};

template <typename T>
struct RngInputs {
  // Meaning of 'start' and 'end' parameter for various distributions
  //
  //         Uniform   Normal/Log-Normal   Gumbel   Logistic   Laplace   Exponential   Rayleigh
  // start    start          mean           mean     mean       mean       lambda       sigma
  // end       end           sigma          beta     scale      scale      Unused       Unused
  T start, end;
  RandomType type;
  uint64_t seed;
};

template <typename T>
class RngPcgHostTest : public ::testing::TestWithParam<RngInputs<T>> {
 public:
  RngPcgHostTest()
    : params(::testing::TestWithParam<RngInputs<T>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      d_buffer(0, stream)
  {
    d_buffer.resize(total_threads*2, stream);
   h_buffer.resize(total_threads*2);
  }

 protected:
  void SetUp() override
  {
    RngState r(params.seed, GenPC);
    DeviceState<PCGenerator> d_state(r);
    switch (params.type) {
      case RNG_Normal:
      {
        constexpr int IPC = 2;
        constexpr size_t len = total_threads * CPT * IPC;
        NormalDistParams<T> normal_params = { .mu = params.start, .sigma = params.end};

        d_buffer.resize(len, stream);
        h_buffer.resize(len);

        pcg_device_kernel<T, NormalDistParams<T>, CPT, IPC><<<n_blocks, n_threads>>>(d_buffer.data(), d_state, normal_params, total_threads, len);
        RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

        for(size_t tid = 0; tid < total_threads; tid++) {
          single_thread_fill<T, NormalDistParams<T>, CPT, IPC>(h_buffer.data(), d_state, normal_params, total_threads, len, tid);
        }
        break;
      }
      case RNG_LogNormal:
        printf("running for lognormal\n");
        break;
      case RNG_Uniform:
      {
        constexpr int IPC = 1;
        constexpr size_t len = total_threads * CPT * IPC;
        printf("len = %lu\n", len);
        UniformDistParams<T> uniform_params = { .start = params.start, .end = params.end};

        d_buffer.resize(len, stream);
        h_buffer.resize(len);

        pcg_device_kernel<T, UniformDistParams<T>, CPT, IPC><<<n_blocks, n_threads>>>(d_buffer.data(), d_state, uniform_params, total_threads, len);
        RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

        for(size_t tid = 0; tid < total_threads; tid++) {
          single_thread_fill<T, UniformDistParams<T>, CPT, IPC>(h_buffer.data(), d_state, uniform_params, total_threads, len, tid);
        }
        break;
      }
      case RNG_Gumbel: printf("Running for gumbel\n"); break;
      case RNG_Logistic:
        printf("running for logistic\n");
        break;
      case RNG_Exp: printf("running for exponential\n"); break;
      case RNG_Rayleigh: printf("running for rayleigh\n"); break;
      case RNG_Laplace:
        printf("running for laplace\n");
        break;
    };
    
  }


 protected:

  static const int n_blocks = 128;
  static const int n_threads = 64;
  static const int CPT = 16;
  static const size_t total_threads = size_t(n_blocks) * n_threads;

  raft::resources handle;
  cudaStream_t stream;

  RngInputs<T> params;
  rmm::device_uvector<T> d_buffer;
  std::vector<T> h_buffer;
};

const std::vector<RngInputs<double>> inputsf = {
  {3.0f, 1.3f, RNG_Normal, 1234ULL},
  {1.2f, 0.1f, RNG_LogNormal, 1234ULL},
  {1.2f, 5.5f, RNG_Uniform, 1234ULL},
  {0.1f, 1.3f, RNG_Gumbel, 1234ULL},
  {1.6f, 0.0f, RNG_Exp, 1234ULL},
  {1.6f, 0.0f, RNG_Rayleigh, 1234ULL},
  {2.6f, 1.3f, RNG_Laplace, 1234ULL}};

using RngPcgHostTestD = RngPcgHostTest<double>;
TEST_P(RngPcgHostTestD, Result) {
  ASSERT_TRUE(devArrMatchHost(h_buffer.data(), d_buffer.data(), 8192, raft::CompareApprox<double>(1e-5), stream));
}
INSTANTIATE_TEST_SUITE_P(RngPcgHostTest, RngPcgHostTestD, testing::ValuesIn(inputsf));


template <typename T>
class TypedTestExample : public testing::Test {
  public:
    void calculate_size() {
      printf("calculate_size called %lu\n", sizeof(params));
    }
    using ParamType = typename T::first_type;
    using DataType = typename T::second_type;
    static ParamType params;
    static DataType d;
    static std::string distro_name;
};

//using TestTypes = testing::Types<char, int, unsigned long, std::pair<NormalDistParams<float>, float>>;
using TestTypes = testing::Types<std::pair<NormalDistParams<float>, float>, std::pair<NormalDistParams<double>, double>>;

TYPED_TEST_SUITE_P(TypedTestExample);

TYPED_TEST_P(TypedTestExample, printSize) {
  this->calculate_size(); 
  printf("Test passed for %s\n", this->distro_name.c_str());
}
REGISTER_TYPED_TEST_SUITE_P(TypedTestExample,
                            printSize);

//template<> float TypedTestExample<char>::start = 1.0;
//template<> float TypedTestExample<int>::start = 2.0;
//template<> float TypedTestExample<unsigned long>::start = 2.0;
template<> NormalDistParams<float> TypedTestExample<std::pair<NormalDistParams<float>, float>>::params = { .mu = 100, .sigma = double(0.1)};
template<> float TypedTestExample<std::pair<NormalDistParams<float>, float>>::d = float(1.2);
template<> std::string TypedTestExample<std::pair<NormalDistParams<float>, float>>::distro_name = std::string("Normal distribution"); 
template<> NormalDistParams<double> TypedTestExample<std::pair<NormalDistParams<double>, double>>::params = { .mu = double(0.5), .sigma = double(0.1)};
template<> double TypedTestExample<std::pair<NormalDistParams<double>, double>>::d = double(1.2);
template<> std::string TypedTestExample<std::pair<NormalDistParams<double>, double>>::distro_name = std::string("Normal double distribution"); 
INSTANTIATE_TYPED_TEST_SUITE_P(My, TypedTestExample, TestTypes);


} // namespace random
} // namespace raft
