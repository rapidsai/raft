/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
#if defined RAFT_COMPILED
#include <raft/distance/specializations.cuh>
#endif

#include <common/benchmark.hpp>
#include <memory>
#include <raft/core/device_resources.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/distance/kernels.cuh>
#include <raft/random/rng.cuh>
#include <sstream>
#include <string>
#include <vector>

namespace raft::bench::distance::kernels {

using namespace raft::distance::kernels;
struct GramTestParams {
  int m;  // m parameter of the GEMM
  int k;  // k parameter of the GEMM
  int n;  // n parameter of the GEMM
  KernelParams kernel_params;
  bool is_row_major;
};  // struct GramTestParams

template <typename T>
struct GramMatrix : public fixture {
  GramMatrix(const GramTestParams& p)
    : params(p), handle(stream), A(0, stream), B(0, stream), C(0, stream)
  {
    kernel = std::unique_ptr<GramMatrixBase<T>>(
      KernelFactory<T>::create(p.kernel_params, handle.get_cublas_handle()));

    A.resize(params.m * params.k, stream);
    B.resize(params.k * params.n, stream);
    C.resize(params.m * params.n, stream);
    raft::random::Rng r(123456ULL);
    r.uniform(A.data(), params.m * params.k, T(-1.0), T(1.0), stream);
    r.uniform(B.data(), params.k * params.n, T(-1.0), T(1.0), stream);
  }

  ~GramMatrix()
  {
    A.release();
    B.release();
    C.release();
  }

  void run_benchmark(::benchmark::State& state) override
  {
    if (!this->kernel) { state.SkipWithError("Kernel matrix is not initialized"); }
    loop_on_state(state, [this]() {
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
  const raft::device_resources handle;
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

RAFT_BENCH_REGISTER(GramMatrix<float>, "", getInputs());
RAFT_BENCH_REGISTER(GramMatrix<double>, "", getInputs());

}  // namespace raft::bench::distance::kernels
