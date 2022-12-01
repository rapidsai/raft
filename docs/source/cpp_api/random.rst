Random
======

This page provides C++ class references for the publicly-exposed elements of the random package.

.. role:: py(code)
   :language: c++
   :class: highlight

#include <raft/random/rng_state.hpp>

.. doxygenstruct:: raft::random::RngState
    :project: RAFT
    :members:


Data Generation
###############

make_blobs
----------

#include *<raft/random/make_blobs.cuh>*

namespace raft::random

.. doxygenfunction:: raft::random::make_blobs(raft::handle_t const& handle, raft::device_matrix_view<DataT, IdxT, layout> out, raft::device_vector_view<IdxT, IdxT> labels, IdxT n_clusters, std::optional<raft::device_matrix_view<DataT, IdxT, layout>> centers, std::optional<raft::device_vector_view<DataT, IdxT>> const cluster_std, const DataT cluster_std_scalar, bool shuffle, DataT center_box_min, DataT center_box_max, uint64_t seed, GeneratorType type)
    :project: RAFT

make_regression
---------------

#include *<raft/random/make_regression.cuh>*

namespace *raft::random*

.. doxygenfunction:: raft::random::make_regression(const raft::handle_t& handle, raft::device_matrix_view<DataT, IdxT, raft::row_major> out, raft::device_matrix_view<DataT, IdxT, raft::row_major> values, IdxT n_informative, std::optional<raft::device_matrix_view<DataT, IdxT, raft::row_major>> coef, DataT bias, IdxT effective_rank, DataT tail_strength, DataT noise, bool shuffle, uint64_t seed, GeneratorType type)
    :project: RAFT

rmat
----

#include *<raft/random/rmat_rectangular_generator.cuh*

namespace *raft::random*

.. doxygenfunction:: raft::random::rmat_rectangular_gen(const raft::handle_t& handle, raft::random::RngState& r, raft::device_vector_view<const ProbT, IdxT> theta, raft::device_mdspan<IdxT, raft::extents<IdxT, raft::dynamic_extent, 2>, raft::row_major> out, raft::device_vector_view<IdxT, IdxT> out_src, raft::device_vector_view<IdxT, IdxT> out_dst, IdxT r_scale, IdxT c_scale)
    :project: RAFT


Random Sampling
###############

Univariate
----------

#include *<raft/random/rng.cuh>*

namespace *raft::random*

.. doxygenfunction:: raft::random::uniform(const raft::handle_t& handle, RngState& rng_state, raft::device_vector_view<OutputValueType, IndexType> out, OutputValueType start, OutputValueType end)
    :project: RAFT

.. doxygenfunction:: raft::random::uniformInt(const raft::handle_t& handle, RngState& rng_state, raft::device_vector_view<OutputValueType, IndexType> out, OutputValueType start, OutputValueType end)
    :project: RAFT

.. doxygenfunction:: raft::random::normal(const raft::handle_t& handle, RngState& rng_state, raft::device_vector_view<OutputValueType, IndexType> out, OutputValueType mu, OutputValueType sigma)
    :project: RAFT

.. doxygenfunction:: raft::random::normalInt(const raft::handle_t& handle, RngState& rng_state, raft::device_vector_view<OutputValueType, IndexType> out, OutputValueType mu, OutputValueType sigma)
    :project: RAFT

.. doxygenfunction:: raft::random::normalTable(const raft::handle_t& handle, RngState& rng_state, raft::device_vector_view<const OutputValueType, IndexType> mu_vec, std::variant<raft::device_vector_view<const OutputValueType, IndexType>, OutputValueType> sigma, raft::device_matrix_view<OutputValueType, IndexType, raft::row_major> out)
    :project: RAFT

.. doxygenfunction:: raft::random::fill(const raft::handle_t& handle, RngState& rng_state, OutputValueType val, raft::device_vector_view<OutputValueType, IndexType> out)
    :project: RAFT

.. doxygenfunction:: raft::random::bernoulli(const raft::handle_t& handle, RngState& rng_state, raft::device_vector_view<OutputValueType, IndexType> out, Type prob)
    :project: RAFT

.. doxygenfunction:: raft::random::scaled_bernoulli(const raft::handle_t& handle, RngState& rng_state, raft::device_vector_view<OutputValueType, IndexType> out, OutputValueType prob, OutputValueType scale)
    :project: RAFT

.. doxygenfunction:: raft::random::gumbel(const raft::handle_t& handle, RngState& rng_state, raft::device_vector_view<OutputValueType, IndexType> out, OutputValueType mu, OutputValueType beta)
    :project: RAFT

.. doxygenfunction:: raft::random::lognormal(const raft::handle_t& handle, RngState& rng_state, raft::device_vector_view<OutputValueType, IndexType> out, OutputValueType mu, OutputValueType sigma)
    :project: RAFT

.. doxygenfunction:: raft::random::logistic(const raft::handle_t& handle, RngState& rng_state, raft::device_vector_view<OutputValueType, IndexType> out, OutputValueType mu, OutputValueType scale)
    :project: RAFT

.. doxygenfunction:: raft::random::exponential(const raft::handle_t& handle, RngState& rng_state, raft::device_vector_view<OutputValueType, IndexType> out, OutputValueType lambda)
    :project: RAFT

.. doxygenfunction:: raft::random::rayleigh(const raft::handle_t& handle, RngState& rng_state, raft::device_vector_view<OutputValueType, IndexType> out, OutputValueType sigma)
    :project: RAFT

.. doxygenfunction:: raft::random::laplace(const raft::handle_t& handle, RngState& rng_state, raft::device_vector_view<OutputValueType, IndexType> out, OutputValueType mu, OutputValueType scale)
    :project: RAFT

.. doxygenfunction:: raft::random::discrete
    :project: RAFT


Multi-Variable Gaussian
-----------------------

#include *<raft/random/multi_variable_gaussian.hpp>*

namespace *raft::random*

.. doxygengroup:: multi_variable_gaussian
    :project: RAFT
    :members:
    :content-only:


Sample Without Replacement
--------------------------

#include *<raft/random/sample_without_replacement.cuh>*

namespace *raft::random*

.. doxygengroup:: sample_without_replacement
    :project: RAFT
    :members:
    :content-only:

#include *<raft/random/permute.cuh>*

namespace *raft::random*

.. doxygengroup:: permute
    :project: RAFT
    :members:
    :content-only:


