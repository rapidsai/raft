/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file thirdparty/LICENSES/LICENSE.faiss
 */

#pragma once

namespace raft::neighbors::detail::faiss_select {
// If the inner size (dim) of the vectors is small, we want a larger query tile
// size, like 1024
inline void chooseTileSize(size_t numQueries,
                           size_t numCentroids,
                           size_t dim,
                           size_t elementSize,
                           size_t totalMem,
                           size_t& tileRows,
                           size_t& tileCols)
{
  // The matrix multiplication should be large enough to be efficient, but if
  // it is too large, we seem to lose efficiency as opposed to
  // double-streaming. Each tile size here defines 1/2 of the memory use due
  // to double streaming. We ignore available temporary memory, as that is
  // adjusted independently by the user and can thus meet these requirements
  // (or not). For <= 4 GB GPUs, prefer 512 MB of usage. For <= 8 GB GPUs,
  // prefer 768 MB of usage. Otherwise, prefer 1 GB of usage.
  size_t targetUsage = 0;

  if (totalMem <= ((size_t)4) * 1024 * 1024 * 1024) {
    targetUsage = 512 * 1024 * 1024;
  } else if (totalMem <= ((size_t)8) * 1024 * 1024 * 1024) {
    targetUsage = 768 * 1024 * 1024;
  } else {
    targetUsage = 1024 * 1024 * 1024;
  }

  targetUsage /= 2 * elementSize;

  // 512 seems to be a batch size sweetspot for float32.
  // If we are on float16, increase to 512.
  // If the k size (vec dim) of the matrix multiplication is small (<= 32),
  // increase to 1024.
  size_t preferredTileRows = 512;
  if (dim <= 32) { preferredTileRows = 1024; }

  tileRows = std::min(preferredTileRows, numQueries);

  // tileCols is the remainder size
  tileCols = std::min(targetUsage / preferredTileRows, numCentroids);
}
}  // namespace raft::neighbors::detail::faiss_select
