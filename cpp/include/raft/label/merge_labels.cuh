/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <math.h>
#include <limits>

#include <raft/cudart_utils.h>
#include <raft/linalg/init.h>
#include <raft/cuda_utils.cuh>

namespace raft {
namespace label {

/** Note: this is one possible implementation where we represent the label
 *  equivalence graph implicitly using labels_a, labels_b and mask.
 *  For an additional cost we can build the graph with edges
 *  E={(A[i], B[i]) | M[i]=1} and make this step faster */
template <typename value_idx, int TPB_X = 256>
__global__ void __launch_bounds__(TPB_X)
  propagate_label_kernel(const value_idx* __restrict__ labels_a,
                         const value_idx* __restrict__ labels_b,
                         value_idx* __restrict__ R,
                         const bool* __restrict__ mask, bool* __restrict__ m,
                         value_idx N) {
  value_idx tid = threadIdx.x + blockIdx.x * TPB_X;
  if (tid < N) {
    if (__ldg((char*)mask + tid)) {
      // Note: labels are from 1 to N
      value_idx la = __ldg(labels_a + tid) - 1;
      value_idx lb = __ldg(labels_b + tid) - 1;
      value_idx ra = R[la];
      value_idx rb = R[lb];
      if (ra != rb) {
        *m = true;
        // min(ra, rb) would be sufficient but this speeds up convergence
        value_idx rmin = R[min(ra, rb)];
        if (sizeof(value_idx) == 4) {
          atomicMin((int*)(R + la), rmin);
          atomicMin((int*)(R + lb), rmin);
        } else if (sizeof(value_idx) == 8) {
          atomicMin((long long int*)(R + la), rmin);
          atomicMin((long long int*)(R + lb), rmin);
        }
      }
    }
  }
}

template <typename value_idx, int TPB_X = 256>
__global__ void __launch_bounds__(TPB_X)
  reassign_label_kernel(value_idx* __restrict__ labels_a,
                        const value_idx* __restrict__ labels_b,
                        const value_idx* __restrict__ R, value_idx N,
                        value_idx MAX_LABEL) {
  value_idx tid = threadIdx.x + blockIdx.x * TPB_X;
  if (tid < N) {
    // Note: labels are from 1 to N
    value_idx la = labels_a[tid];
    value_idx lb = __ldg(labels_b + tid);
    value_idx ra = (la == MAX_LABEL) ? MAX_LABEL : __ldg(R + (la - 1)) + 1;
    value_idx rb = (lb == MAX_LABEL) ? MAX_LABEL : __ldg(R + (lb - 1)) + 1;
    labels_a[tid] = min(ra, rb);
  }
}

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
void merge_labels(value_idx* labels_a, const value_idx* labels_b,
                  const bool* mask, value_idx* R, bool* m, value_idx N,
                  cudaStream_t stream) {
  dim3 blocks(raft::ceildiv(N, value_idx(TPB_X)));
  dim3 threads(TPB_X);
  value_idx MAX_LABEL = std::numeric_limits<value_idx>::max();

  // Initialize R. R defines the relabeling rules; after merging the input
  // arrays, label l will be reassigned as R[l-1]+1.
  raft::linalg::range(R, N, stream);

  // We define the label equivalence graph: G = (V, E), where:
  //  - V is the set of unique values from labels_a and labels_b
  //  - E = {(labels_a[k], labels_b[k]) | mask[k] == true and k \in 0..n-1 }
  // The edges connect groups from the two labellings. Only points with true
  // mask can induce connection between groups.

  // Step 1: compute connected components in the label equivalence graph
  bool host_m;
  do {
    CUDA_CHECK(cudaMemsetAsync(m, false, sizeof(bool), stream));

    propagate_label_kernel<value_idx, TPB_X>
      <<<blocks, threads, 0, stream>>>(labels_a, labels_b, R, mask, m, N);
    CUDA_CHECK(cudaPeekAtLastError());

    raft::update_host(&host_m, m, 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  } while (host_m);

  // Step 2: re-assign minimum equivalent label
  reassign_label_kernel<value_idx, TPB_X>
    <<<blocks, threads, 0, stream>>>(labels_a, labels_b, R, N, MAX_LABEL);
  CUDA_CHECK(cudaPeekAtLastError());
}

};  // namespace label
};  // namespace raft