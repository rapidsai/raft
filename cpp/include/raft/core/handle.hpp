/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_resources.hpp>

namespace raft {

/**
 * raft::handle_t is being kept around for backwards
 * compatibility and will be removed in a future version.
 *
 * Extending the `raft::handle_t` instead of `using` to
 * minimize needed changes downstream
 * (e.g. existing forward declarations, etc...)
 *
 * Use of `raft::resources` or `raft::handle_t` is preferred.
 */
class handle_t : public raft::device_resources {
 public:
  handle_t(const handle_t& handle) : device_resources{handle} {}

  handle_t(handle_t&&)            = delete;
  handle_t& operator=(handle_t&&) = delete;

  /**
   * @brief Construct a resources instance with a stream view and stream pool
   *
   * @param[in] stream_view the default stream (which has the default per-thread stream if
   * unspecified)
   * @param[in] stream_pool the stream pool used (which has default of nullptr if unspecified)
   */
  handle_t(rmm::cuda_stream_view stream_view                  = rmm::cuda_stream_per_thread,
           std::shared_ptr<rmm::cuda_stream_pool> stream_pool = {nullptr})
    : device_resources{stream_view, stream_pool}
  {
  }

  /** Destroys all held-up resources */
  ~handle_t() override {}
};

}  // end NAMESPACE raft
