#=============================================================================
# Copyright (c) 2021-2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

# keep the files in alphabetical order!
add_executable(test_raft
#    test/cluster/kmeans.cu
    test/common/logger.cpp
    test/common/seive.cu
    test/cudart_utils.cpp
#    test/cluster_solvers.cu
    test/cluster_solvers_deprecated.cu
    test/device_atomics.cu
    test/distance/dist_adj.cu
    test/distance/dist_canberra.cu
    test/distance/dist_chebyshev.cu
    test/distance/dist_correlation.cu
    test/distance/dist_cos.cu
    test/distance/dist_euc_exp.cu
    test/distance/dist_euc_unexp.cu
    test/distance/dist_hamming.cu
    test/distance/dist_hellinger.cu
    test/distance/dist_jensen_shannon.cu
    test/distance/dist_kl_divergence.cu
    test/distance/dist_l1.cu
    test/distance/dist_minkowski.cu
    test/distance/dist_russell_rao.cu
    test/distance/fused_l2_nn.cu
    test/eigen_solvers.cu
    test/handle.cpp
    test/integer_utils.cpp
    test/interruptible.cu
    test/nvtx.cpp
    test/pow2_utils.cu
    test/label/label.cu
    test/label/merge_labels.cu
    test/lap/lap.cu
    test/linalg/add.cu
    test/linalg/binary_op.cu
    test/linalg/cholesky_r1.cu
    test/linalg/coalesced_reduction.cu
    test/linalg/divide.cu
    test/linalg/eig.cu
    test/linalg/eig_sel.cu
    test/linalg/gemm_layout.cu
    test/linalg/gemv.cu
    test/linalg/map.cu
    test/linalg/map_then_reduce.cu
    test/linalg/matrix_vector_op.cu
    test/linalg/multiply.cu
    test/linalg/norm.cu
    test/linalg/power.cu
    test/linalg/reduce.cu
    test/linalg/reduce_cols_by_key.cu
    test/linalg/reduce_rows_by_key.cu
    test/linalg/rsvd.cu
    test/linalg/sqrt.cu
    test/linalg/strided_reduction.cu
    test/linalg/subtract.cu
    test/linalg/svd.cu
    test/linalg/ternary_op.cu
    test/linalg/transpose.cu
    test/linalg/unary_op.cu
    test/matrix/gather.cu
    test/matrix/math.cu
    test/matrix/matrix.cu
    test/matrix/columnSort.cu
    test/matrix/linewise_op.cu
    test/mdarray.cu
    test/mdspan_utils.cu
    test/mst.cu
    test/random/make_blobs.cu
    test/random/make_regression.cu
    test/random/multi_variable_gaussian.cu
    test/random/permute.cu
    test/random/rng.cu
    test/random/rng_int.cu
    test/random/rmat_rectangular_generator.cu
    test/random/sample_without_replacement.cu
    test/span.cpp
    test/span.cu
    test/sparse/add.cu
    test/sparse/convert_coo.cu
    test/sparse/convert_csr.cu
    test/sparse/connect_components.cu
    test/sparse/csr_row_slice.cu
    test/sparse/csr_to_dense.cu
    test/sparse/csr_transpose.cu
    test/sparse/degree.cu
    test/sparse/dist_coo_spmv.cu
    test/sparse/distance.cu
    test/sparse/filter.cu
    test/sparse/knn.cu
    test/sparse/knn_graph.cu
    test/sparse/linkage.cu
    test/sparse/norm.cu
    test/sparse/reduce.cu
    test/sparse/row_op.cu
    test/sparse/sort.cu
    test/sparse/symmetrize.cu
    test/spatial/ann_ivf_flat.cu
    test/spatial/knn.cu
    test/spatial/fused_l2_knn.cu
    test/spatial/haversine.cu
    test/spatial/ball_cover.cu
    test/spatial/epsilon_neighborhood.cu
    test/spatial/faiss_mr.cu
    test/spatial/selection.cu
    test/spectral_matrix.cu
    test/stats/adjusted_rand_index.cu
    test/stats/completeness_score.cu
    test/stats/contingencyMatrix.cu
    test/stats/cov.cu
    test/stats/dispersion.cu
    test/stats/entropy.cu
    test/stats/histogram.cu
    test/stats/homogeneity_score.cu
    test/stats/information_criterion.cu
    test/stats/kl_divergence.cu
    test/stats/mean.cu
    test/stats/meanvar.cu
    test/stats/mean_center.cu
    test/stats/minmax.cu
    test/stats/mutual_info_score.cu
    test/stats/rand_index.cu
    test/stats/silhouette_score.cu
    test/stats/stddev.cu
    test/stats/sum.cu
    test/stats/trustworthiness.cu
    test/stats/weighted_mean.cu
    test/stats/v_measure.cu
    test/test.cpp
)

set_target_properties(test_raft
        PROPERTIES BUILD_RPATH                         "\$ORIGIN"
        # set target compile options
        CXX_STANDARD                        17
        CXX_STANDARD_REQUIRED               ON
        CUDA_STANDARD                       17
        CUDA_STANDARD_REQUIRED              ON
        POSITION_INDEPENDENT_CODE           ON
        INTERFACE_POSITION_INDEPENDENT_CODE ON
        INSTALL_RPATH "\$ORIGIN/../../../lib"
        )

target_compile_options(test_raft
        PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAFT_CXX_FLAGS}>"
        "$<$<COMPILE_LANGUAGE:CUDA>:${RAFT_CUDA_FLAGS}>"
        )

target_include_directories(test_raft
        PUBLIC  "$<BUILD_INTERFACE:${RAFT_SOURCE_DIR}/test>"
        )


target_link_libraries(test_raft
        PRIVATE
        raft::raft
        raft::distance
        raft::nn
        GTest::gtest
        GTest::gtest_main
        Threads::Threads
        $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
        $<TARGET_NAME_IF_EXISTS:conda_env>)

if(SKBUILD)
        install(
                TARGETS test_raft
                COMPONENT testing
                DESTINATION bin/libraft/gtests
        )
else()
        install(
                TARGETS test_raft
                COMPONENT testing
                DESTINATION bin/libraft/gtests
                EXCLUDE_FROM_ALL
        )
endif()
