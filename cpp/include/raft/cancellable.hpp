/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <raft/cudart_utils.h>
#include <mutex>
#include <raft/error.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <thread>
#include <unordered_map>

namespace raft {

/**
 * @brief Exception thrown during `cancellable::synchronize` call when it detects a request
 * to cancel the work performed in this CPU thread.
 */
struct cancelled : public raft::exception {
  explicit cancelled(char const* const message) : raft::exception(message) {}
  explicit cancelled(std::string const& message) : raft::exception(message) {}
};

class cancellable {
 public:
  /**
   * @brief Synchronize the CUDA stream, subject to being cancelled by `cancellable::cancel`
   * called on this CPU thread.
   *
   * @param [in] stream a CUDA stream.
   *
   * @throw raft::cancelled if cancellable::cancel() was called on the current CPU thread id
   * before the currently captured work has been finished.
   * @throw raft::cuda_error if another CUDA error happens.
   */
  static void synchronize(rmm::cuda_stream_view stream) { store_.synchronize(stream); }

  /**
   * @brief Cancel any current or next call to `cancellable::synchronize` performed on the
   * CPU thread given by the `thread_id`
   *
   * @param [in] thread_id a CPU thread, in which the work should be cancelled.
   *
   * @throw raft::cuda_error if a CUDA error happens during recording the interruption event.
   */
  static void cancel(std::thread::id thread_id) { store::cancel(thread_id); }

 private:
  /*
   * Implementation-wise, the cancellation feature is bound to the CPU threads.
   * Each thread has an associated thread-local state store comprising a boolean flag
   * and a CUDA event. The event plays the role of a condition variable; it can be triggered
   * either when the work captured in the stream+event is finished or when the cancellation request
   * is issued to the current CPU thread. For the latter, we keep internally one global stream
   * for recording the "interrupt" events in any CPU thread.
   */
  static inline thread_local class store {
   private:
    /** Global registery of thread-local cancellation stores. */
    static inline std::unordered_map<std::thread::id, store*> registry_;
    /** Protect the access to the registry. */
    static inline std::mutex mutex_;
    /** The only purpose of this stream is to record the interruption events. */
    static inline std::unique_ptr<cudaStream_t, std::function<void(cudaStream_t*)>>
      cancellation_stream_{
        []() {
          auto* stream = new cudaStream_t;
          RAFT_CUDA_TRY(cudaStreamCreateWithFlags(stream, cudaStreamNonBlocking));
          return stream;
        }(),
        [](cudaStream_t* stream) {
          RAFT_CUDA_TRY(cudaStreamDestroy(*stream));
          delete stream;
        }};

    /** The state of being in the process of cancelling. */
    bool cancelled_ = false;
    /** The main synchronization primitive for the current CPU thread on the CUDA side.  */
    cudaEvent_t wait_interrupt_ = nullptr;

   public:
    store()
    {
      std::lock_guard<std::mutex> guard(mutex_);
      registry_[std::this_thread::get_id()] = this;
      RAFT_CUDA_TRY(
        cudaEventCreateWithFlags(&wait_interrupt_, cudaEventBlockingSync | cudaEventDisableTiming));
    }
    ~store()
    {
      std::lock_guard<std::mutex> guard(mutex_);
      registry_.erase(std::this_thread::get_id());
      cudaEventDestroy(wait_interrupt_);
    }

    void synchronize(rmm::cuda_stream_view stream)
    {
      // This function synchronizes the CPU thread on the "interrupt" event instead of
      // the given stream.
      // Assuming that this method is called only on a thread-local store, there is no need for
      // extra synchronization primitives to protect the state.
      RAFT_CUDA_TRY(cudaEventRecord(wait_interrupt_, stream));
      RAFT_CUDA_TRY(cudaEventSynchronize(wait_interrupt_));
      if (cancelled_) {
        cancelled_ = false;
        throw cancelled("The work was in this stream was cancelled.");
      }
    }

    void cancel()
    {
      // This method is supposed to be called from another thread;
      // multiple calls to it just override each other, and that is ok - the cancellation request
      // will be delivered (at least once).
      cancelled_ = true;
      RAFT_CUDA_TRY(cudaEventRecord(wait_interrupt_, *cancellation_stream_));
    }

    static void cancel(std::thread::id thread_id)
    {
      // The mutex here is neededd to make sure the registry_ is not accessed during
      // the registration of a new thread (when the registry_ is altered).
      std::lock_guard<std::mutex> guard(mutex_);
      registry_[thread_id]->cancel();
    }
  } store_;
};

}  // namespace raft
