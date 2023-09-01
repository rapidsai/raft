# =============================================================================
# Copyright (c) 2021-2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.
# =============================================================================

# ##################################################################################################
# enable testing ################################################################################
# ##################################################################################################
enable_testing()
include(rapids-test)
rapids_test_init()

function(ConfigureTest)

  set(options OPTIONAL LIB EXPLICIT_INSTANTIATE_ONLY)
  set(oneValueArgs NAME GPUS PERCENT)
  set(multiValueArgs PATH TARGETS CONFIGURATIONS)

  cmake_parse_arguments(_RAFT_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(NOT DEFINED _RAFT_TEST_GPUS AND NOT DEFINED _RAFT_TEST_PERCENT)
    set(_RAFT_TEST_GPUS 1)
    set(_RAFT_TEST_PERCENT 30)
  endif()
  if(NOT DEFINED _RAFT_TEST_GPUS)
    set(_RAFT_TEST_GPUS 1)
  endif()
  if(NOT DEFINED _RAFT_TEST_PERCENT)
    set(_RAFT_TEST_PERCENT 100)
  endif()

  set(TEST_NAME ${_RAFT_TEST_NAME})

  add_executable(${TEST_NAME} ${_RAFT_TEST_PATH})
  target_link_libraries(
    ${TEST_NAME}
    PRIVATE raft
            raft_internal
            $<$<BOOL:${_RAFT_TEST_LIB}>:raft::compiled>
            GTest::gtest
            GTest::gtest_main
            Threads::Threads
            ${RAFT_CTK_MATH_DEPENDENCIES}
            $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
            $<TARGET_NAME_IF_EXISTS:conda_env>
  )
  set_target_properties(
    ${TEST_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${RAFT_BINARY_DIR}/gtests>"
               INSTALL_RPATH "\$ORIGIN/../../../lib"
               CXX_STANDARD 17
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 17
               CUDA_STANDARD_REQUIRED ON
  )
  target_compile_options(
    ${TEST_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAFT_CXX_FLAGS}>"
                         "$<$<COMPILE_LANGUAGE:CUDA>:${RAFT_CUDA_FLAGS}>"
  )
  if(_RAFT_TEST_EXPLICIT_INSTANTIATE_ONLY)
    target_compile_definitions(${TEST_NAME} PRIVATE "RAFT_EXPLICIT_INSTANTIATE_ONLY")
  endif()

  target_include_directories(${TEST_NAME} PUBLIC "$<BUILD_INTERFACE:${RAFT_SOURCE_DIR}/test>")

  rapids_test_add(
    NAME ${TEST_NAME}
    COMMAND ${TEST_NAME}
    GPUS ${_RAFT_TEST_GPUS}
    PERCENT ${_RAFT_TEST_PERCENT}
    INSTALL_COMPONENT_SET testing
  )
endfunction()

# ##################################################################################################
# test sources ##################################################################################
# ##################################################################################################

# ##################################################################################################
# * distance tests -------------------------------------------------------------------------

if(BUILD_TESTS)
  ConfigureTest(
    NAME
    CLUSTER_TEST
    PATH
    test/cluster/kmeans.cu
    test/cluster/kmeans_balanced.cu
    test/cluster/cluster_solvers.cu
    test/cluster/linkage.cu
    test/cluster/kmeans_find_k.cu
    LIB
    EXPLICIT_INSTANTIATE_ONLY
  )

  ConfigureTest(
    NAME
    CORE_TEST
    PATH
    test/core/device_resources_manager.cpp
    test/core/device_setter.cpp
    test/core/logger.cpp
    test/core/math_device.cu
    test/core/math_host.cpp
    test/core/operators_device.cu
    test/core/operators_host.cpp
    test/core/handle.cpp
    test/core/interruptible.cu
    test/core/nvtx.cpp
    test/core/mdarray.cu
    test/core/mdspan_utils.cu
    test/core/numpy_serializer.cu
    test/core/memory_type.cpp
    test/core/sparse_matrix.cu
    test/core/sparse_matrix.cpp
    test/core/span.cpp
    test/core/span.cu
    test/core/temporary_device_buffer.cu
    test/test.cpp
    LIB
    EXPLICIT_INSTANTIATE_ONLY
  )

  ConfigureTest(
    NAME
    DISTANCE_TEST
    PATH
    test/distance/dist_adj.cu
    test/distance/dist_adj_distance_instance.cu
    test/distance/dist_canberra.cu
    test/distance/dist_correlation.cu
    test/distance/dist_cos.cu
    test/distance/dist_hamming.cu
    test/distance/dist_hellinger.cu
    test/distance/dist_inner_product.cu
    test/distance/dist_jensen_shannon.cu
    test/distance/dist_kl_divergence.cu
    test/distance/dist_l1.cu
    test/distance/dist_l2_exp.cu
    test/distance/dist_l2_unexp.cu
    test/distance/dist_l2_sqrt_exp.cu
    test/distance/dist_l_inf.cu
    test/distance/dist_lp_unexp.cu
    test/distance/dist_russell_rao.cu
    test/distance/masked_nn.cu
    test/distance/masked_nn_compress_to_bits.cu
    test/distance/fused_l2_nn.cu
    test/distance/gram.cu
    LIB
    EXPLICIT_INSTANTIATE_ONLY
  )

  list(
    APPEND
    EXT_HEADER_TEST_SOURCES
    test/ext_headers/raft_neighbors_brute_force.cu
    test/ext_headers/raft_distance_distance.cu
    test/ext_headers/raft_distance_detail_pairwise_matrix_dispatch.cu
    test/ext_headers/raft_matrix_detail_select_k.cu
    test/ext_headers/raft_neighbors_ball_cover.cu
    test/ext_headers/raft_spatial_knn_detail_fused_l2_knn.cu
    test/ext_headers/raft_distance_fused_l2_nn.cu
    test/ext_headers/raft_neighbors_ivf_pq.cu
    test/ext_headers/raft_util_memory_pool.cpp
    test/ext_headers/raft_neighbors_ivf_flat.cu
    test/ext_headers/raft_core_logger.cpp
    test/ext_headers/raft_neighbors_refine.cu
    test/ext_headers/raft_neighbors_detail_ivf_flat_search.cu
    test/ext_headers/raft_neighbors_detail_selection_faiss.cu
    test/ext_headers/raft_linalg_detail_coalesced_reduction.cu
    test/ext_headers/raft_spatial_knn_detail_ball_cover_registers.cu
    test/ext_headers/raft_neighbors_detail_ivf_flat_interleaved_scan.cu
    test/ext_headers/raft_neighbors_detail_ivf_pq_compute_similarity.cu
  )

  # Test that the split headers compile in isolation with:
  #
  # * EXT_HEADERS_TEST_COMPILED_EXPLICIT: RAFT_COMPILED, RAFT_EXPLICIT_INSTANTIATE_ONLY defined
  # * EXT_HEADERS_TEST_COMPILED_IMPLICIT: RAFT_COMPILED defined
  # * EXT_HEADERS_TEST_IMPLICIT:          no macros defined.
  ConfigureTest(
    NAME EXT_HEADERS_TEST_COMPILED_EXPLICIT PATH ${EXT_HEADER_TEST_SOURCES} LIB
    EXPLICIT_INSTANTIATE_ONLY
  )
  ConfigureTest(NAME EXT_HEADERS_TEST_COMPILED_IMPLICIT PATH ${EXT_HEADER_TEST_SOURCES} LIB)
  ConfigureTest(NAME EXT_HEADERS_TEST_IMPLICIT PATH ${EXT_HEADER_TEST_SOURCES})

  ConfigureTest(NAME LABEL_TEST PATH test/label/label.cu test/label/merge_labels.cu)

  ConfigureTest(
    NAME
    LINALG_TEST
    PATH
    test/linalg/add.cu
    test/linalg/axpy.cu
    test/linalg/binary_op.cu
    test/linalg/cholesky_r1.cu
    test/linalg/coalesced_reduction.cu
    test/linalg/divide.cu
    test/linalg/dot.cu
    test/linalg/eig.cu
    test/linalg/eig_sel.cu
    test/linalg/gemm_layout.cu
    test/linalg/gemv.cu
    test/linalg/map.cu
    test/linalg/map_then_reduce.cu
    test/linalg/matrix_vector.cu
    test/linalg/matrix_vector_op.cu
    test/linalg/mean_squared_error.cu
    test/linalg/multiply.cu
    test/linalg/norm.cu
    test/linalg/normalize.cu
    test/linalg/power.cu
    test/linalg/randomized_svd.cu
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
  )

  ConfigureTest(
    NAME
    MATRIX_TEST
    PATH
    test/matrix/argmax.cu
    test/matrix/argmin.cu
    test/matrix/columnSort.cu
    test/matrix/diagonal.cu
    test/matrix/gather.cu
    test/matrix/scatter.cu
    test/matrix/eye.cu
    test/matrix/linewise_op.cu
    test/matrix/math.cu
    test/matrix/matrix.cu
    test/matrix/norm.cu
    test/matrix/reverse.cu
    test/matrix/slice.cu
    test/matrix/triangular.cu
    test/sparse/spectral_matrix.cu
    LIB
    EXPLICIT_INSTANTIATE_ONLY
  )

  ConfigureTest(NAME MATRIX_SELECT_TEST PATH test/matrix/select_k.cu LIB EXPLICIT_INSTANTIATE_ONLY)

  ConfigureTest(
    NAME MATRIX_SELECT_LARGE_TEST PATH test/matrix/select_large_k.cu LIB EXPLICIT_INSTANTIATE_ONLY
  )

  ConfigureTest(
    NAME
    RANDOM_TEST
    PATH
    test/random/make_blobs.cu
    test/random/make_regression.cu
    test/random/multi_variable_gaussian.cu
    test/random/rng_pcg_host_api.cu
    test/random/permute.cu
    test/random/rng.cu
    test/random/rng_discrete.cu
    test/random/rng_int.cu
    test/random/rmat_rectangular_generator.cu
    test/random/sample_without_replacement.cu
  )

  ConfigureTest(
    NAME SOLVERS_TEST PATH test/cluster/cluster_solvers_deprecated.cu test/linalg/eigen_solvers.cu
    test/lap/lap.cu test/sparse/mst.cu LIB EXPLICIT_INSTANTIATE_ONLY
  )

  ConfigureTest(
    NAME
    SPARSE_TEST
    PATH
    test/sparse/add.cu
    test/sparse/convert_coo.cu
    test/sparse/convert_csr.cu
    test/sparse/csr_row_slice.cu
    test/sparse/csr_to_dense.cu
    test/sparse/csr_transpose.cu
    test/sparse/degree.cu
    test/sparse/filter.cu
    test/sparse/norm.cu
    test/sparse/normalize.cu
    test/sparse/reduce.cu
    test/sparse/row_op.cu
    test/sparse/sort.cu
    test/sparse/spgemmi.cu
    test/sparse/symmetrize.cu
  )

  ConfigureTest(
    NAME SPARSE_DIST_TEST PATH test/sparse/dist_coo_spmv.cu test/sparse/distance.cu
    test/sparse/gram.cu LIB EXPLICIT_INSTANTIATE_ONLY
  )

  ConfigureTest(
    NAME
    SPARSE_NEIGHBORS_TEST
    PATH
    test/sparse/neighbors/cross_component_nn.cu
    test/sparse/neighbors/brute_force.cu
    test/sparse/neighbors/knn_graph.cu
    LIB
    EXPLICIT_INSTANTIATE_ONLY
  )

  ConfigureTest(
    NAME
    NEIGHBORS_TEST
    PATH
    test/neighbors/knn.cu
    test/neighbors/fused_l2_knn.cu
    test/neighbors/tiled_knn.cu
    test/neighbors/haversine.cu
    test/neighbors/ball_cover.cu
    test/neighbors/epsilon_neighborhood.cu
    test/neighbors/refine.cu
    LIB
    EXPLICIT_INSTANTIATE_ONLY
  )

  ConfigureTest(
    NAME
    NEIGHBORS_ANN_CAGRA_TEST
    PATH
    test/neighbors/ann_cagra/test_float_uint32_t.cu
    test/neighbors/ann_cagra/test_int8_t_uint32_t.cu
    test/neighbors/ann_cagra/test_uint8_t_uint32_t.cu
    test/neighbors/ann_cagra/test_float_int64_t.cu
    src/neighbors/detail/cagra/search_multi_cta_float_uint64_dim128_t8.cu
    src/neighbors/detail/cagra/search_multi_cta_float_uint64_dim256_t16.cu
    src/neighbors/detail/cagra/search_multi_cta_float_uint64_dim512_t32.cu
    src/neighbors/detail/cagra/search_multi_cta_float_uint64_dim1024_t32.cu
    src/neighbors/detail/cagra/search_single_cta_float_uint64_dim128_t8.cu
    src/neighbors/detail/cagra/search_single_cta_float_uint64_dim256_t16.cu
    src/neighbors/detail/cagra/search_single_cta_float_uint64_dim512_t32.cu
    src/neighbors/detail/cagra/search_single_cta_float_uint64_dim1024_t32.cu
    LIB
    EXPLICIT_INSTANTIATE_ONLY
    GPUS
    1
    PERCENT
    100
  )

  ConfigureTest(
    NAME
    NEIGHBORS_ANN_IVF_TEST
    PATH
    test/neighbors/ann_ivf_flat/test_float_int64_t.cu
    test/neighbors/ann_ivf_flat/test_int8_t_int64_t.cu
    test/neighbors/ann_ivf_flat/test_uint8_t_int64_t.cu
    test/neighbors/ann_ivf_pq/test_float_int64_t.cu
    test/neighbors/ann_ivf_pq/test_float_uint32_t.cu
    test/neighbors/ann_ivf_pq/test_float_int64_t.cu
    test/neighbors/ann_ivf_pq/test_int8_t_int64_t.cu
    test/neighbors/ann_ivf_pq/test_uint8_t_int64_t.cu
    LIB
    EXPLICIT_INSTANTIATE_ONLY
    GPUS
    1
    PERCENT
    100
  )

  ConfigureTest(
    NAME NEIGHBORS_SELECTION_TEST PATH test/neighbors/selection.cu LIB EXPLICIT_INSTANTIATE_ONLY
    GPUS 1 PERCENT 50
  )

  ConfigureTest(
    NAME
    STATS_TEST
    PATH
    test/stats/accuracy.cu
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
    test/stats/r2_score.cu
    test/stats/rand_index.cu
    test/stats/regression_metrics.cu
    test/stats/silhouette_score.cu
    test/stats/stddev.cu
    test/stats/sum.cu
    test/stats/trustworthiness.cu
    test/stats/weighted_mean.cu
    test/stats/v_measure.cu
    LIB
    EXPLICIT_INSTANTIATE_ONLY
  )

  ConfigureTest(
    NAME
    UTILS_TEST
    PATH
    test/core/seive.cu
    test/util/bitonic_sort.cu
    test/util/cudart_utils.cpp
    test/util/device_atomics.cu
    test/util/integer_utils.cpp
    test/util/integer_utils.cu
    test/util/pow2_utils.cu
    test/util/reduction.cu
  )
endif()

# ##################################################################################################
# Install tests ####################################################################################
# ##################################################################################################
rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing DESTINATION bin/gtests/libraft)
