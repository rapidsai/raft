/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include <mutex>
#include <raft/cudart_utils.h>
#include <raft/error.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <thread>
#include <unordered_map>

namespace raft {

/**
 * @brief Exception thrown during `interruptible::synchronize` call when it detects a request
 * to cancel the work performed in this CPU thread.
 */
struct interrupted_exception : public raft::exception {
  using raft::exception::exception;
};

class interruptible {
 public:
  /**
   * @brief Synchronize the CUDA stream, subject to being interrupted by `interruptible::cancel`
   * called on this CPU thread.
   *
   * @param [in] stream a CUDA stream.
   *
   * @throw raft::interrupted_exception if interruptible::cancel() was called on the current CPU
   * thread id before the currently captured work has been finished.
   * @throw raft::cuda_error if another CUDA error happens.
   */
  static inline void synchronize(rmm::cuda_stream_view stream)
  {
    get_token()->synchronize_impl(stream);
  }

  /**
   * @brief Check the thread state, whether the thread is interrupted by `interruptible::cancel`.
   *
   * This is a cancellation point for an interruptible thread. It's called in the internals of
   * `interruptible::synchronize` in a loop. If two synchronize calls are far apart, it's
   * recommended to call `interruptible::yield()` in between to make sure the thread does not become
   * unresponsive for too long.
   *
   * Both `yield` and `yield_no_throw` reset the state to non-cancelled after execution.
   *
   * @throw raft::interrupted_exception if interruptible::cancel() was called on the current CPU
   * thread.
   */
  static inline void yield() { get_token()->yield_impl(); }

  /**
   * @brief Check the thread state, whether the thread is interrupted by `interruptible::cancel`.
   *
   * Same as `interruptible::yield`, but does not throw an exception if the thread is cancelled.
   *
   * Both `yield` and `yield_no_throw` reset the state to non-cancelled after execution.
   *
   * @return whether the thread can continue, i.e. `true` means continue, `false` means cancelled.
   */
  static inline auto yield_no_throw() -> bool { return get_token()->yield_no_throw_impl(); }

  /**
   * @brief Get a cancellation token for this CPU thread.
   *
   * @return an object that can be used to cancel the GPU work waited on this CPU thread.
   */
  static inline auto get_token() -> std::shared_ptr<interruptible>
  {
    static thread_local std::shared_ptr<interruptible> s(get_token(std::this_thread::get_id()));
    return s;
  }

  /**
   * @brief Get a cancellation token for a CPU thread given by its id.
   *
   * @return an object that can be used to cancel the GPU work waited on the given CPU thread.
   */
  static auto get_token(std::thread::id thread_id) -> std::shared_ptr<interruptible>
  {
    std::lock_guard<std::mutex> guard_get(mutex_);
    // the following constructs an empty shared_ptr if the key does not exist.
    auto& weak_store  = registry_[thread_id];
    auto thread_store = weak_store.lock();
    if (!thread_store) {
      thread_store.reset(new interruptible(), [thread_id](auto ts) {
        std::lock_guard<std::mutex> guard_erase(mutex_);
        registry_.erase(thread_id);
        delete ts;
      });
      std::weak_ptr<interruptible>(thread_store).swap(weak_store);
    }
    return thread_store;
  }

  /**
   * @brief Cancel any current or next call to `interruptible::synchronize` performed on the
   * CPU thread given by the `thread_id`
   *
   * Note, this function uses a mutex to safely get a cancellation token that may be shared
   * among multiple threads. If you plan to use it from a signal handler, consider the non-static
   * `cancel()` instead.
   *
   * @param [in] thread_id a CPU thread, in which the work should be interrupted.
   */
  static inline void cancel(std::thread::id thread_id) { get_token(thread_id)->cancel(); }

  /**
   * @brief Cancel any current or next call to `interruptible::synchronize` performed on the
   * CPU thread given by this `interruptible` token.
   *
   * Note, this function does not involve thread synchronization/locks and does not throw any
   * exceptions, so it's safe to call from a signal handler.
   */
  inline void cancel() noexcept { continue_.clear(std::memory_order_relaxed); }

  // don't allow the token to leave the shared_ptr
  interruptible(interruptible const&) = delete;
  interruptible(interruptible&&)      = delete;
  auto operator=(interruptible const&) -> interruptible& = delete;
  auto operator=(interruptible&&) -> interruptible& = delete;

 private:
  /** Global registry of thread-local cancellation stores. */
  static inline std::unordered_map<std::thread::id, std::weak_ptr<interruptible>> registry_;
  /** Protect the access to the registry. */
  static inline std::mutex mutex_;

  /**
   * Communicate whether the thread is in a cancelled state or can continue execution.
   *
   * `yield` checks this flag and always resets it to the signalled state; `cancel` clears it.
   * These are the only two places where it's used.
   */
  std::atomic_flag continue_;

  interruptible() noexcept { yield_no_throw_impl(); }

  void yield_impl()
  {
    if (!yield_no_throw_impl()) {
      throw interrupted_exception("The work in this thread was cancelled.");
    }
  }

  auto yield_no_throw_impl() noexcept -> bool
  {
    return continue_.test_and_set(std::memory_order_relaxed);
  }

  void synchronize_impl(rmm::cuda_stream_view stream)
  {
    cudaError_t query_result;
    while (true) {
      yield_impl();
      query_result = cudaStreamQuery(stream);
      if (query_result != cudaErrorNotReady) { break; }
      std::this_thread::yield();
    }
    RAFT_CUDA_TRY(query_result);
  }
};

}  // namespace raft
