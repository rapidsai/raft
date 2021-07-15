#include <gtest/gtest.h>

#include <raft/iterator_traits.hpp>

namespace raft {

TEST(Raft, IteratorTraits) {
  static_assert(is_random_access_iterator<int *>::value, "working");
  static_assert(is_random_access_iterator<int const *>::value, "working");
  static_assert(is_random_access_iterator<int *const>::value, "working");
  static_assert(is_random_access_iterator<int const *const>::value, "working");
}

}  // namespace raft