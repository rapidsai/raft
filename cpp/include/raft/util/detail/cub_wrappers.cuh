/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <rmm/device_uvector.hpp>

#include <cub/cub.cuh>

namespace raft {

/**
 * @brief Convenience wrapper over cub's SortPairs method
 * @tparam KeyT key type
 * @tparam ValueT value type
 * @param workspace workspace buffer which will get resized if not enough space
 * @param inKeys input keys array
 * @param outKeys output keys array
 * @param inVals input values array
 * @param outVals output values array
 * @param len array length
 * @param stream cuda stream
 */
template <typename KeyT, typename ValueT>
void sortPairs(rmm::device_uvector<char>& workspace,
               const KeyT* inKeys,
               KeyT* outKeys,
               const ValueT* inVals,
               ValueT* outVals,
               int len,
               cudaStream_t stream)
{
  size_t worksize = 0;  //  Fix 'worksize' may be used uninitialized in this function.
  cub::DeviceRadixSort::SortPairs(
    nullptr, worksize, inKeys, outKeys, inVals, outVals, len, 0, sizeof(KeyT) * 8, stream);
  workspace.resize(worksize, stream);
  cub::DeviceRadixSort::SortPairs(
    workspace.data(), worksize, inKeys, outKeys, inVals, outVals, len, 0, sizeof(KeyT) * 8, stream);
}

}  // namespace raft
