#include <gtest/gtest.h>

#include <raft/iterator_traits.hpp>

#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace raft {

TEST(Raft, PointerIteratorTraits) {
  static_assert(is_random_access_buffer<int *>::value);
  static_assert(is_random_access_buffer<int const *>::value);
  static_assert(is_random_access_buffer<int *const>::value);
  static_assert(is_random_access_buffer<int const *const>::value);
}

TEST(Raft, ThrustIteratorTraits) {
  static_assert(
    is_random_access_buffer<thrust::device_vector<int>::iterator>::value);

  static_assert(
    is_random_access_buffer<thrust::device_vector<const int>::iterator>::value);

  auto transform_op = [] __host__ __device__(const auto &x) { return x; };

  static_assert(
    is_random_access_buffer<
      thrust::transform_iterator<decltype(transform_op), int *>>::value);

  static_assert(
    is_random_access_buffer<thrust::counting_iterator<const int>>::value);
  static_assert(
    is_random_access_buffer<const thrust::constant_iterator<int>>::value);
}

}  // namespace raft