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

namespace raft {
namespace detail {
auto static constexpr const TRANSPOSE_TILE_DIM = 32;

template <typename OutType, typename InType, typename IndexType>
__global__ void transpose(
  OutType* out,
  InType* in,
  IndexType in_major_dim,
  IndexType in_minor_dim
) {
  __shared__ OutType tile[TRANSPOSE_TILE_DIM][TRANSPOSE_TILE_DIM + 1];
  auto static constexpr const TILE_ELEMENTS = (
    TRANSPOSE_TILE_DIM * TRANSPOSE_TILE_DIM
  );
  auto const max_index = in_major_dim * in_minor_dim;

  for (auto i=0; i < max_index; i += TILE_ELEMENTS) {
    auto in_x = blockIdx.x * TRANSPOSE_TILE_DIM + threadIdx.x;
    auto in_y = blockIdx.y * TRANSPOSE_TILE_DIM + threadIdx.y;
    tile[in_x][in_y] = static_cast<OutType>(in[in_major * in_x + in_y]);
  }
}

} // namespace detail
} // namespace raft
