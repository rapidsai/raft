/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#ifndef __MERGE_LABELS_H
#define __MERGE_LABELS_H

#pragma once

#include <raft/label/detail/merge_labels.cuh>

namespace raft {
namespace label {

/**
 * @brief Merge two labellings in-place, according to a core mask
 *
 * A labelling is a representation of disjoint sets (groups) where points that
 * belong to the same group have the same label. It is assumed that group
 * labels take values between 1 and N. labels relate to points, i.e a label i+1
 * means that you belong to the same group as the point i.
 * The special value MAX_LABEL is used to mark points that are not labelled.
 *
 * The two label arrays A and B induce two sets of groups over points 0..N-1.
 * If a point is labelled i in A and j in B and the mask is true for this
 * point, then i and j are equivalent labels and their groups are merged by
 * relabeling the elements of both groups to have the same label. The new label
 * is the smaller one from the original labels.
 * It is required that if the mask is true for a point, this point is labelled
 * (i.e its label is different than the special value MAX_LABEL).
 *
 * One use case is finding connected components: the two input label arrays can
 * represent the connected components of graphs G_A and G_B, and the output
 * would be the connected components labels of G_A \union G_B.
 *
 * @param[inout] labels_a    First input, and output label array (in-place)
 * @param[in]    labels_b    Second input label array
 * @param[in]    mask        Core point mask
 * @param[out]   R           label equivalence map
 * @param[in]    m           Working flag
 * @param[in]    N           Number of points in the dataset
 * @param[in]    stream      CUDA stream
 */
template <typename value_idx = int, int TPB_X = 256>
void merge_labels(value_idx* labels_a,
                  const value_idx* labels_b,
                  const bool* mask,
                  value_idx* R,
                  bool* m,
                  value_idx N,
                  cudaStream_t stream)
{
  detail::merge_labels<value_idx, TPB_X>(labels_a, labels_b, mask, R, m, N, stream);
}

};  // namespace label
};  // namespace raft

#endif