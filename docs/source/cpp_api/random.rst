Random
======

This page provides C++ class references for the publicly-exposed elements of the random package.

.. role:: py(code)
   :language: c++
   :class: highlight


.. doxygenstruct:: raft::random::RngState
    :project: RAFT
    :members:


Data Generation
###############

.. doxygenfunction:: raft::random::make_blobs(raft::handle_t const& handle, raft::device_matrix_view<DataT, IdxT, layout> out, raft::device_vector_view<IdxT, IdxT> labels, IdxT n_clusters, std::optional<raft::device_matrix_view<DataT, IdxT, layout>> centers, std::optional<raft::device_vector_view<DataT, IdxT>> const cluster_std, const DataT cluster_std_scalar, bool shuffle, DataT center_box_min, DataT center_box_max, uint64_t seed, GeneratorType type)
    :project: RAFT

.. doxygenfunction:: raft::random::make_regression(const raft::handle_t& handle, raft::device_matrix_view<DataT, IdxT, raft::row_major> out, raft::device_matrix_view<DataT, IdxT, raft::row_major> values, IdxT n_informative, std::optional<raft::device_matrix_view<DataT, IdxT, raft::row_major>> coef, DataT bias, IdxT effective_rank, DataT tail_strength, DataT noise, bool shuffle, uint64_t seed, GeneratorType type)
    :project: RAFT

.. doxygenfunction:: raft::random::rmat_rectangular_gen(const raft::handle_t& handle, raft::random::RngState& r, raft::device_vector_view<const ProbT, IdxT> theta, raft::device_mdspan<IdxT, raft::extents<IdxT, raft::dynamic_extent, 2>, raft::row_major> out, raft::device_vector_view<IdxT, IdxT> out_src, raft::device_vector_view<IdxT, IdxT> out_dst, IdxT r_scale, IdxT c_scale)
    :project: RAFT


Random Sampling
###############

.. doxygenfunction:: raft::random::uniform(const raft::handle_t& handle, RngState& rng_state, raft::device_vector_view<OutputValueType, IndexType> out, OutputValueType start, OutputValueType end)

.. doxygenfunction:: raft::random::uniformInt(const raft::handle_t& handle, RngState& rng_state, raft::device_vector_view<OutputValueType, IndexType> out, OutputValueType start, OutputValueType end)

.. doxygenfunction:: raft::random::normal(const raft::handle_t& handle, RngState& rng_state, raft::device_vector_view<OutputValueType, IndexType> out, OutputValueType mu, OutputValueType sigma)

.. doxygenfunction:: raft::random::normalInt(const raft::handle_t& handle, RngState& rng_state, raft::device_vector_view<OutputValueType, IndexType> out, OutputValueType mu, OutputValueType sigma)

.. doxygenfunction:: raft::random::normalTable(const raft::handle_t& handle, RngState& rng_state, raft::device_vector_view<const OutputValueType, IndexType> mu_vec, std::variant<raft::device_vector_view<const OutputValueType, IndexType>, OutputValueType> sigma, raft::device_matrix_view<OutputValueType, IndexType, raft::row_major> out)

.. doxygenfunction:: raft::random::fill(const raft::handle_t& handle, RngState& rng_state, OutputValueType val, raft::device_vector_view<OutputValueType, IndexType> out)

.. doxygenfunction:: raft::random::bernoulli(const raft::handle_t& handle, RngState& rng_state, raft::device_vector_view<OutputValueType, IndexType> out, Type prob)

.. doxygenfunction:: raft::random::scaled_bernoulli(const raft::handle_t& handle, RngState& rng_state, raft::device_vector_view<OutputValueType, IndexType> out, OutputValueType prob, OutputValueType scale)

.. doxygenfunction:: raft::random::gumbel(const raft::handle_t& handle, RngState& rng_state, raft::device_vector_view<OutputValueType, IndexType> out, OutputValueType mu, OutputValueType beta)

.. doxygenfunction:: raft::random::lognormal(const raft::handle_t& handle, RngState& rng_state, raft::device_vector_view<OutputValueType, IndexType> out, OutputValueType mu, OutputValueType sigma)

.. doxygenfunction:: raft::random::logistic(const raft::handle_t& handle, RngState& rng_state, raft::device_vector_view<OutputValueType, IndexType> out, OutputValueType mu, OutputValueType scale)

.. doxygenfunction:: raft::random::exponential(const raft::handle_t& handle, RngState& rng_state, raft::device_vector_view<OutputValueType, IndexType> out, OutputValueType lambda)

.. doxygenfunction:: raft::random::rayleigh(const raft::handle_t& handle, RngState& rng_state, raft::device_vector_view<OutputValueType, IndexType> out, OutputValueType sigma)

.. doxygenfunction:: raft::random::laplace(const raft::handle_t& handle, RngState& rng_state, raft::device_vector_view<OutputValueType, IndexType> out, OutputValueType mu, OutputValueType scale)

.. doxygenfunction:: raft::random::sample_without_replacement


Useful Operations
#################

.. doxygenfunction:: raft::random::permute(const raft::handle_t& handle, raft::device_matrix_view<const InputOutputValueType, IdxType, Layout> in, std::optional<raft::device_vector_view<IntType, IdxType>> permsOut, std::optional<raft::device_matrix_view<InputOutputValueType, IdxType, Layout>> out)
    :project: RAFT


