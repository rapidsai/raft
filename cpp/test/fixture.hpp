/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <gtest/gtest.h>
#include <raft/handle.hpp>
#include <raft/cudart_utils.h>

namespace raft {

template <typename ParamT>
class fixture : public ::testing::TestWithParam<ParamT> {
 protected:
  virtual void initialize() {
  }

  virtual void finalize() {
  }

  virtual void check() {
  }

  void SetUp() override {  // NOLINT
    CUDA_CHECK(cudaStreamCreate(&stream_));
    handle_.reset(new raft::handle_t);
    handle_->set_stream(stream_);
    initialize();
    CUDA_CHECK(cudaDeviceSynchronize());  // to be safe
  }

  void TearDown() override {  // NOLINT
    CUDA_CHECK(cudaDeviceSynchronize());  // to be safe
    finalize();
    handle_.reset();
    CUDA_CHECK(cudaStreamDestroy(stream_));
  }

  const raft::handle_t& handle() const { return *handle_; }

  std::shared_ptr<raft::handle_t> handle_;
  cudaStream_t stream_;
};  // class fixture


#define RUN_TEST(test_suite_name, test_name, test_type, inputs) \
  using test_name = test_type;                                  \
                                                                \
  TEST_P(test_name, Result) {                                   \
    this->check();                                              \
  }                                                             \
                                                                \
  INSTANTIATE_TEST_SUITE_P(test_suite_name, test_name,          \
    ::testing::ValuesIn(inputs))

}  // namespace raft
