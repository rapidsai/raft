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

#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <omp.h>
#include <raft/common/nvtx.hpp>
#include <raft/interruptible.hpp>
#include <rmm/cuda_stream.hpp>
#include <thread>
#include <vector>

namespace raft {

__global__ void gpu_wait(int millis)
{
  for (auto i = millis; i > 0; i--) {
    __nanosleep(1000000);
  }
}

TEST(Raft, InterruptibleBasic)
{
  ASSERT_FALSE(interruptible::yield_no_throw());

  // Cancel using the token
  interruptible::get_token()->cancel();
  ASSERT_TRUE(interruptible::yield_no_throw());
  ASSERT_FALSE(interruptible::yield_no_throw());

  // Cancel by thread id
  interruptible::cancel(std::this_thread::get_id());
  ASSERT_TRUE(interruptible::yield_no_throw());
  ASSERT_FALSE(interruptible::yield_no_throw());
}

TEST(Raft, InterruptibleOpenMP)
{
  // number of threads must be smaller than max number of resident grids for GPU
  const int n_threads = 10;
  // 1 <= n_expected_succeed <= n_threads
  const int n_expected_succeed = 5;
  // How many milliseconds passes between a thread i and i+1 finishes.
  // i.e. thread i executes (C + i*n_expected_succeed) milliseconds in total.
  const int thread_delay_millis = 20;
  common::nvtx::range fun_scope("interruptible");

  std::vector<std::shared_ptr<interruptible>> thread_tokens(n_threads);
  int n_finished  = 0;
  int n_cancelled = 0;

  omp_set_dynamic(0);
  omp_set_num_threads(n_threads);
#pragma omp parallel reduction(+ : n_finished) reduction(+ : n_cancelled) num_threads(n_threads)
  {
    auto i = omp_get_thread_num();
    common::nvtx::range omp_scope("interruptible::thread-%d", i);
    rmm::cuda_stream stream;
    gpu_wait<<<1, 1, 0, stream.value()>>>(1);
    interruptible::synchronize(stream);
    thread_tokens[i] = interruptible::get_token();

#pragma omp barrier
    try {
      common::nvtx::range wait_scope("interruptible::wait-%d", i);
      gpu_wait<<<1, 1, 0, stream.value()>>>((1 + i) * thread_delay_millis);
      interruptible::synchronize(stream);
      n_finished = 1;
    } catch (interrupted_exception&) {
      n_cancelled = 1;
    }
    if (i == n_expected_succeed - 1) {
      common::nvtx::range cancel_scope("interruptible::cancel-%d", i);
      for (auto token : thread_tokens)
        token->cancel();
    }

#pragma omp barrier
    // clear the cancellation state to not disrupt other tests
    interruptible::yield_no_throw();
  }
  ASSERT_EQ(n_finished, n_expected_succeed);
  ASSERT_EQ(n_cancelled, n_threads - n_expected_succeed);
}
}  // namespace raft
