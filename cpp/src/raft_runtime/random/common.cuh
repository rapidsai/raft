/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rmat_rectangular_generator.cuh>

#include <raft_runtime/random/rmat_rectangular_generator.hpp>

#define FUNC_DEF(IdxT, ProbT)                                                          \
  void rmat_rectangular_gen(raft::resources const& handle,                             \
                            IdxT* out,                                                 \
                            IdxT* out_src,                                             \
                            IdxT* out_dst,                                             \
                            const ProbT* theta,                                        \
                            IdxT r_scale,                                              \
                            IdxT c_scale,                                              \
                            IdxT n_edges,                                              \
                            raft::random::RngState& r)                                 \
  {                                                                                    \
    raft::random::rmat_rectangular_gen<IdxT, ProbT>(out,                               \
                                                    out_src,                           \
                                                    out_dst,                           \
                                                    theta,                             \
                                                    r_scale,                           \
                                                    c_scale,                           \
                                                    n_edges,                           \
                                                    resource::get_cuda_stream(handle), \
                                                    r);                                \
  }
