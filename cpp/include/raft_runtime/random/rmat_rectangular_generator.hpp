/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
