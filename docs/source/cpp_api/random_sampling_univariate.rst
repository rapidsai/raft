Univariate Random Sampling
==========================

.. role:: py(code)
   :language: c++
   :class: highlight

``#include <raft/random/rng.cuh>``

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


