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
  explicit interrupted_exception(char const* const message) : raft::exception(message) {}
  explicit interrupted_exception(std::string const& message) : raft::exception(message) {}
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
   * @return whether interruptible::cancel() was called on the current CPU thread.
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
      // Since we're still guarded by the mutex, use this opportunity to initialize the global
      // cancellation stream. The cancellation stream is only used by non-static cancel(), so it's
      // safe to initialize it here before the first valid thread store is returned to the user.
      get_cancellation_stream();
    }
    return thread_store;
  }

  /**
   * @brief Cancel any current or next call to `interruptible::synchronize` performed on the
   * CPU thread given by the `thread_id`
   *
   * Note, this function uses a mutex to safely get a cancellation token that may be shared
   * among multiple threads. If you plan to use it from a signal handler, consider the non-static
   * `cancel_no_throw()` instead.
   *
   * @param [in] thread_id a CPU thread, in which the work should be interrupted.
   *
   * @throw raft::cuda_error if a CUDA error happens during recording the interruption event.
   */
  static inline void cancel(std::thread::id thread_id) { get_token(thread_id)->cancel(); }

  /**
   * @brief Cancel any current or next call to `interruptible::synchronize` performed on the
   * CPU thread given by this `interruptible` token.
   *
   * Note, this function does not involve thread synchronization/locks, so it should be safe
   * to call from a signal handler.
   *
   * @throw raft::cuda_error if a CUDA error happens during recording the interruption event.
   */
  inline void cancel() { RAFT_CUDA_TRY(cancel_no_throw()); }

  /**
   * @brief Cancel any current or next call to `interruptible::synchronize` performed on the
   * CPU thread given by this `interruptible` token.
   *
   * Note, this function does not involve thread synchronization/locks and not throw any
   * exceptions, so it's safe to call from a signal handler.
   *
   * @return CUDA error code redirected from `cudaEventRecord`.
   */
  auto cancel_no_throw() noexcept -> cudaError_t
  {
    // This method is supposed to be called from another thread;
    // multiple calls to it just override each other, and that is ok - the cancellation request
    // will be delivered (at least once).
    cancelled_ = true;
    return cudaEventRecord(wait_interrupt_, get_cancellation_stream());
  }

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
  /** The only purpose of this stream is to record the interruption events. */
  static inline auto get_cancellation_stream() -> cudaStream_t
  {
    static std::unique_ptr<cudaStream_t, std::function<void(cudaStream_t*)>> cs{
      []() {
        auto* stream = new cudaStream_t;
        RAFT_CUDA_TRY(cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking));
        return stream;
      }(),
      [](cudaStream_t* stream) {
        cudaStreamDestroy(*stream);
        delete stream;
      }};
    return *cs;
  }

  /*
   * Implementation-wise, the cancellation feature is bound to the CPU threads.
   * Each thread has an associated thread-local state store comprising a boolean flag
   * and a CUDA event. The event plays the role of a condition variable; it can be triggered
   * either when the work captured in the stream+event is finished or when the cancellation request
   * is issued to the current CPU thread. For the latter, we keep internally one global stream
   * for recording the "interrupt" events in any CPU thread.
   */

  /** The state of being in the process of cancelling. */
  bool cancelled_ = false;

  /** The main synchronization primitive for the current CPU thread on the CUDA side.  */
  cudaEvent_t wait_interrupt_ = nullptr;

  interruptible()
  {
    /*
    NOTE: it would be nice to have `cudaEventBlockingSync` flag here for better CPU utilization in
    a multi-threaded setting. However, it may occasionally lead to a significant latency when
    many streams/threads are operating concurrently.

    The problem can be ovserved in test/interruptible.cu::Raft.InterruptibleOpenMP launched using
    nsys. Some of the CPU threads may stuck in `sem_wait` for a few hundred milliseconds.
     */
    RAFT_CUDA_TRY(cudaEventCreateWithFlags(&wait_interrupt_, cudaEventDisableTiming));
  }

  ~interruptible() { cudaEventDestroy(wait_interrupt_); }

  void yield_impl()
  {
    if (yield_no_throw_impl()) {
      throw interrupted_exception("The work in this thread was cancelled.");
    }
  }

  auto yield_no_throw_impl() noexcept -> bool
  {
    if (cancelled_) {
      cancelled_ = false;
      return true;
    }
    return false;
  }

  void synchronize_impl(rmm::cuda_stream_view stream)
  {
    // This function synchronizes the CPU thread on the "interrupt" event instead of
    // the given stream.
    // Assuming that this method is called only on a thread-local store, there is no need for
    // extra synchronization primitives to protect the state.
    RAFT_CUDA_TRY(cudaEventRecord(wait_interrupt_, stream));
    RAFT_CUDA_TRY(cudaEventSynchronize(wait_interrupt_));
    yield_impl();
  }
};

}  // namespace raft
