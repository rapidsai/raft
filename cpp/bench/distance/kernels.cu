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
#if defined RAFT_DISTANCE_COMPILED
#include <raft/distance/specializations.cuh>
#endif

#include <common/benchmark.hpp>
#include <memory>
#include <raft/core/handle.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/distance/kernels.cuh>
#include <raft/random/rng.cuh>
#include <sstream>
#include <string>
#include <vector>

namespace raft::distance::kernels::bench {

struct GramTestParams {
  int m;  // m parameter of the GEMM
  int k;  // k parameter of the GEMM
  int n;  // n parameter of the GEMM
  KernelParams kernel_params;
  bool is_row_major;
};  // struct GramTestParams

template <typename T>
struct GramMatrix : public Fixture {
  GramMatrix(const std::string& name, const GramTestParams& p)
    : Fixture(name), params(p), A(0, stream), B(0, stream), C(0, stream)
  {
    std::vector<std::string> kernel_names{"linear", "poly", "rbf", "tanh"};
    std::ostringstream oss;
    oss << name << "/" << kernel_names[p.kernel_params.kernel] << "/" << p.m << "x" << p.k << "x"
        << p.n << "/" << (p.is_row_major ? "row_major" : "col_major");
    this->SetName(oss.str().c_str());

    kernel = std::unique_ptr<GramMatrixBase<T>>(
      KernelFactory<T>::create(p.kernel_params, handle.get_cublas_handle()));
  }

  ~GramMatrix() {}

 protected:
  void allocateBuffers(const ::benchmark::State& state) override
  {
    A.resize(params.m * params.k, stream);
    B.resize(params.k * params.n, stream);
    C.resize(params.m * params.n, stream);
    raft::random::Rng r(123456ULL);
    r.uniform(A.data(), params.m * params.k, T(-1.0), T(1.0), stream);
    r.uniform(B.data(), params.k * params.n, T(-1.0), T(1.0), stream);
  }
  void deallocateBuffers(const ::benchmark::State& state) override
  {
    A.release();
    B.release();
    C.release();
  }
  void runBenchmark(::benchmark::State& state) override
  {
    if (!this->kernel) { state.SkipWithError("Kernel matrix is not initialized"); }
    loopOnState(state, [this]() {
      (*this->kernel)(A.data(),
                      this->params.m,
                      this->params.k,
                      B.data(),
                      this->params.n,
                      C.data(),
                      this->params.is_row_major,
                      this->stream);
    });
  }

 private:
  raft::handle_t& handle;
  std::unique_ptr<GramMatrixBase<T>> kernel;
  GramTestParams params;

  rmm::device_uvector<T> A;  // input matrix A, size [m * k]
  rmm::device_uvector<T> B;  // input matrix B, size [n * k]
  rmm::device_uvector<T> C;  // output matrix C, size [m*n]
};

static std::vector<GramTestParams> getInputs()
{
  std::vector<GramTestParams> param_vec;
  std::vector<KernelParams> kernel_params{KernelParams{LINEAR, 3, 1, 0},
                                          KernelParams{POLYNOMIAL, 2, 1.3, 1},
                                          KernelParams{TANH, 2, 0.5, 2.4},
                                          KernelParams{RBF, 2, 0.5, 0}};
  struct TestSize {
    int m;
    int k;
    int n;
  };
  std::vector<TestSize> data_size{{4096, 10, 1024},
                                  {4096, 100, 1024},
                                  {4096, 1000, 1024},
                                  {4096, 10000, 1024},
                                  {100000, 10, 1024},
                                  {100000, 100, 1024},
                                  {100000, 1000, 1024}};

  param_vec.reserve(kernel_params.size() * data_size.size());
  for (TestSize s : data_size) {
    for (auto kernel : kernel_params) {
      for (bool row_major : {false, true}) {
        param_vec.push_back(GramTestParams{s.m, s.k, s.n, kernel, row_major});
      }
    }
  }
  return param_vec;
}

ML_BENCH_REGISTER(GramTestParams, GramMatrix<float>, "", getInputs());
ML_BENCH_REGISTER(GramTestParams, GramMatrix<double>, "", getInputs());

}  // namespace raft::distance::kernels::bench
