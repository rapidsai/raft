#include <type_traits>

namespace raft {

template <typename T, typename = void>
struct has_access_operator : std::false_type {};

template <typename T>
struct has_access_operator<T, std::void_t<decltype(std::declval<T>()[0])>>
  : std::true_type {};

template <typename T, typename = void>
struct is_random_access_iterator : std::false_type {};

template <typename T>
struct is_random_access_iterator<
  T, std::enable_if_t<has_access_operator<T>::value and std::is_pointer_v<T>>>
  : std::true_type {};

}  // namespace raft