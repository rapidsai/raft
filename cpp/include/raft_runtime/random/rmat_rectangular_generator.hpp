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

#include <raft/core/resources.hpp>
#include <raft/random/rng_state.hpp>

#include <cstdint>

namespace raft::runtime::random {

/**
 * @defgroup rmat_runtime RMAT Runtime API
 * @{
 */

#define FUNC_DECL(IdxT, ProbT)                             \
  void rmat_rectangular_gen(raft::resources const& handle, \
                            IdxT* out,                     \
                            IdxT* out_src,                 \
                            IdxT* out_dst,                 \
                            const ProbT* theta,            \
                            IdxT r_scale,                  \
                            IdxT c_scale,                  \
                            IdxT n_edges,                  \
                            raft::random::RngState& r)

FUNC_DECL(int, float);
FUNC_DECL(int64_t, float);
FUNC_DECL(int, double);
FUNC_DECL(int64_t, double);

#undef FUNC_DECL

/** @} */  // end group rmat_runtime

}  // namespace raft::runtime::random
