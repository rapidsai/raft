/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuda_runtime.h>

namespace raft::resource {

class cuda_event_resource : public resource {
 public:
  cuda_event_resource()
  {
    RAFT_CUDA_TRY_NO_THROW(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
  }
  void* get_resource() override { return &event_; }

  ~cuda_event_resource() override { RAFT_CUDA_TRY_NO_THROW(cudaEventDestroy(event_)); }

 private:
  cudaEvent_t event_;
};
}  // namespace raft::resource
