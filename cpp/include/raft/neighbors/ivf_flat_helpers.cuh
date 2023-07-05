/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/neighbors/detail/ivf_flat_build.cuh>
#include <raft/neighbors/ivf_flat_types.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

namespace raft::neighbors::ivf_flat::helpers {
/**
 * @defgroup ivf_flat_helpers Helper functions for manipulationg IVF Flat Index
 * @{
 */

template <typename T>
void unpackInterleaved(
        const T* in,
        T* out,
        int numVecs,
        int dim,
        int veclen) {

  // The data is written in interleaved groups of `index::kGroupSize` vectors
  using interleaved_group = Pow2<kIndexGroupSize>;

  // Interleave dimensions of the source vector while recording it.
  // NB: such `veclen` is selected, that `dim % veclen == 0`
  #pragma omp parallel for
  for (int i = 0; i < numVecs; i++) {
    auto group_offset = interleaved_group::roundDown(i);
    auto ingroup_id = interleaved_group::mod(i) * veclen;

    // Point to the location of the interleaved group of vectors
    out += group_offset * dim;
    for (uint32_t l = 0; l < dim; l += veclen) {
      for (uint32_t j = 0; j < veclen; j++) {
        out[l * kIndexGroupSize + ingroup_id + j] = in[i * dim + l + j];
      }
    }
  }
}


template <typename T>
void pack_host_interleaved(
        const T* in,
        T* out,
        int numVecs,
        int dim,
        int veclen) {

  // The data is written in interleaved groups of `index::kGroupSize` vectors
  using interleaved_group = Pow2<kIndexGroupSize>;

  // Interleave dimensions of the source vector while recording it.
  // NB: such `veclen` is selected, that `dim % veclen == 0`
  #pragma omp parallel for
  for (int i = 0; i < numVecs; i++) {
    auto group_offset = interleaved_group::roundDown(i);
    auto ingroup_id = interleaved_group::mod(i) * veclen;

    // Point to the location of the interleaved group of vectors
    out += group_offset * dim;
    for (uint32_t l = 0; l < dim; l += veclen) {
      for (uint32_t j = 0; j < veclen; j++) {
        out[l * kIndexGroupSize + ingroup_id + j] = in[i * dim + l + j];
      }
    }
  }
}

template <typename T>
void unpack_host_interleaved(
        const T* in,
        T* out,
        int numVecs,
        int dim,
        int veclen) {

  // The data is written in interleaved groups of `index::kGroupSize` vectors
  using interleaved_group = Pow2<kIndexGroupSize>;

  // Interleave dimensions of the source vector while recording it.
  // NB: such `veclen` is selected, that `dim % veclen == 0`
  #pragma omp parallel for
  for (int i = 0; i < numVecs; i++) {
    auto group_offset = interleaved_group::roundDown(i);
    auto ingroup_id = interleaved_group::mod(i) * veclen;

    // Point to the location of the interleaved group of vectors
    out += group_offset * dim;
    for (uint32_t l = 0; l < dim; l += veclen) {
      for (uint32_t j = 0; j < veclen; j++) {
        out[i * dim + l + j] = in[l * kIndexGroupSize + ingroup_id + j];
      }
    }
  }
}
/** @} */
}  // namespace raft::neighbors::ivf_flat::helpers
