/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#pragma once

#include <memory>

#include <raft/cudart_utils.h>
#include <raft/interruptible.hpp>

#include <benchmark/benchmark.h>

#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

namespace raft::bench {

/**
 * RAII way to temporary set the pooling memory allocator in rmm.
 * This may be useful for benchmarking functions that do some memory allocations.
 */
struct using_pool_memory_res {
 private:
  rmm::mr::device_memory_resource* orig_res_;
  rmm::mr::cuda_memory_resource cuda_res_;
  rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource> pool_res_;

 public:
  using_pool_memory_res(size_t initial_size, size_t max_size)
    : orig_res_(rmm::mr::get_current_device_resource()),
      pool_res_(&cuda_res_, initial_size, max_size)
  {
    rmm::mr::set_current_device_resource(&pool_res_);
  }

  using_pool_memory_res() : using_pool_memory_res(size_t(1) << size_t(30), size_t(16) << size_t(30))
  {
  }

  ~using_pool_memory_res() { rmm::mr::set_current_device_resource(orig_res_); }
};

/**
 * RAII way of timing cuda calls. This has been shamelessly copied from the
 * cudf codebase via cuml codebase. So, credits for this class goes to cudf developers.
 */
struct cuda_event_timer {
 private:
  ::benchmark::State* state_;
  rmm::cuda_stream_view stream_;
  cudaEvent_t start_;
  cudaEvent_t stop_;

 public:
  /**
   * @param state  the benchmark::State whose timer we are going to update.
   * @param stream CUDA stream we are measuring time on.
   */
  cuda_event_timer(::benchmark::State& state, rmm::cuda_stream_view stream)
    : state_(&state), stream_(stream)
  {
    RAFT_CUDA_TRY(cudaEventCreate(&start_));
    RAFT_CUDA_TRY(cudaEventCreate(&stop_));
    raft::interruptible::synchronize(stream_);
    RAFT_CUDA_TRY(cudaEventRecord(start_, stream_));
  }
  cuda_event_timer() = delete;

  /**
   * @brief The dtor stops the timer and performs a synchroniazation. Time of
   *       the benchmark::State object provided to the ctor will be set to the
   *       value given by `cudaEventElapsedTime()`.
   */
  ~cuda_event_timer()
  {
    RAFT_CUDA_TRY_NO_THROW(cudaEventRecord(stop_, stream_));
    raft::interruptible::synchronize(stop_);
    float milliseconds = 0.0f;
    RAFT_CUDA_TRY_NO_THROW(cudaEventElapsedTime(&milliseconds, start_, stop_));
    state_->SetIterationTime(milliseconds / 1000.f);
    RAFT_CUDA_TRY_NO_THROW(cudaEventDestroy(start_));
    RAFT_CUDA_TRY_NO_THROW(cudaEventDestroy(stop_));
  }
};

/** Main fixture to be inherited and used by all other c++ benchmarks */
class fixture {
 private:
  rmm::cuda_stream stream_owner_{};
  rmm::device_buffer scratch_buf_;

 public:
  rmm::cuda_stream_view stream;

  fixture() : stream{stream_owner_.view()}
  {
    int l2_cache_size = 0;
    int device_id     = 0;
    RAFT_CUDA_TRY(cudaGetDevice(&device_id));
    RAFT_CUDA_TRY(cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, device_id));
    scratch_buf_ = rmm::device_buffer(l2_cache_size, stream);
  }

  // every benchmark should be overriding this
  virtual void run_benchmark(::benchmark::State& state) = 0;
  virtual void generate_metrics(::benchmark::State& state) {}

  /**
   * The helper to be used inside `run_benchmark`, to loop over the state and record time using the
   * cuda_event_timer.
   */
  template <typename Lambda>
  void loop_on_state(::benchmark::State& state, Lambda benchmark_func, bool flush_L2 = true)
  {
    for (auto _ : state) {
      if (flush_L2) {
        RAFT_CUDA_TRY(cudaMemsetAsync(scratch_buf_.data(), 0, scratch_buf_.size(), stream));
      }
      cuda_event_timer timer(state, stream);
      benchmark_func();
    }
  }
};

namespace internal {

template <typename Class, typename... Params>
class Fixture : public ::benchmark::Fixture {
  using State = ::benchmark::State;

 public:
  explicit Fixture(const std::string name, const Params&... params)
    : ::benchmark::Fixture(), params_(params...)
  {
    SetName(name.c_str());
  }
  Fixture() = delete;

  void SetUp(const State& state) override
  {
    fixture_ =
      std::apply([](const Params&... ps) { return std::make_unique<Class>(ps...); }, params_);
  }
  void TearDown(const State& state) override { fixture_.reset(); }
  void SetUp(State& st) override { SetUp(const_cast<const State&>(st)); }
  void TearDown(State& st) override { TearDown(const_cast<const State&>(st)); }

 private:
  std::unique_ptr<Class> fixture_;
  std::tuple<Params...> params_;

 protected:
  void BenchmarkCase(State& state) override
  {
    fixture_->run_benchmark(state);
    fixture_->generate_metrics(state);
  }
};  // class Fixture

/**
 * A helper struct to create a fixture for every combination of input vectors.
 * Use with care, this can blow up quickly!
 */
template <typename Class, typename... Params>
struct cartesian_registrar {
  template <typename... Fixed>
  static void run(const std::string case_name,
                  const std::vector<Params>&... params,
                  const Fixed&... fixed);
};

template <typename Class>
struct cartesian_registrar<Class> {
  template <typename... Fixed>
  static void run(const std::string case_name, const Fixed&... fixed)
  {
    auto* b = ::benchmark::internal::RegisterBenchmarkInternal(
      new Fixture<Class, Fixed...>(case_name, fixed...));
    b->UseManualTime();
    b->Unit(benchmark::kMillisecond);
  }
};

template <typename Class, typename Param, typename... Params>
struct cartesian_registrar<Class, Param, Params...> {
  template <typename... Fixed>
  static void run(const std::string case_name,
                  const std::vector<Param>& param,
                  const std::vector<Params>&... params,
                  const Fixed&... fixed)
  {
    int param_len = param.size();
    for (int i = 0; i < param_len; i++) {
      cartesian_registrar<Class, Params...>::run(
        case_name + "/" + std::to_string(i), params..., fixed..., param[i]);
    }
  }
};

template <typename Class>
struct registrar {
  /**
   * Register a fixture `Class` named `testClass` for every combination of input `params`.
   *
   * @param test_class
   *     A string representation of the `Class` name.
   * @param test_name
   *     Optional test name. Leave empty, if you don't need it.
   * @param params
   *     Zero or more vectors of parameters.
   *     The generated test cases are a cartesian product of these vectors.
   *     Use with care, this can blow up quickly!
   */
  template <typename... Params>
  registrar(const std::string& test_class,
            const std::string& test_name,
            const std::vector<Params>&... params)
  {
    std::stringstream name_stream;
    name_stream << test_class;
    if (!test_name.empty()) { name_stream << "/" << test_name; }
    cartesian_registrar<Class, Params...>::run(name_stream.str(), params...);
  }
};

};  // namespace internal

/**
 * This is the entry point macro for all benchmarks. This needs to be called
 * for the set of benchmarks to be registered so that the main harness inside
 * google bench can find these benchmarks and run them.
 *
 * @param TestClass   child class of `raft::bench::Fixture` which contains
 *                    the logic to generate the dataset and run training on it
 *                    for a given algo. Ideally, once such struct is needed for
 *                    every algo to be benchmarked
 * @param test_name   a unique string to identify these tests at the end of run
 *                    This is optional and if choose not to use this, pass an
 *                    empty string
 * @param params...   zero or more lists of params upon which to benchmark.
 */
#define RAFT_BENCH_REGISTER(TestClass, ...)                                             \
  static raft::bench::internal::registrar<TestClass> BENCHMARK_PRIVATE_NAME(registrar)( \
    #TestClass, __VA_ARGS__)

}  // namespace raft::bench
