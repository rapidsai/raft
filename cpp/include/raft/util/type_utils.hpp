/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <variant>

namespace raft {

template <typename variant1, typename variant2>
struct concatenated_variant;

template <typename ... types1, typename ... types2>
struct concatenated_variant <std::variant<types1...>, std::variant<types2...>>{
  using type = std::variant<types1..., types2...>;
};

template<typename variant1, typename variant2>
using concatenated_variant_t = typename concatenated_variant<variant1, variant2>::type;

template <typename visitor_t, typename variant_t, std::size_t index=std::size_t{}>
auto fast_visit (visitor_t&& visitor, variant_t&& variant) {
  using return_t = decltype(
    std::forward<visitor_t>(visitor)(std::get<index>(std::forward<variant_t>(variant))));
  auto result = return_t{};

  if (index == variant.index()) {
    if (!std::holds_alternative<std::variant_alternative_t<index, variant_t>>(variant)) {
      __builtin_unreachable();
    }
    result = std::forward<visitor_t>(visitor)(std::get<index>(std::forward<variant_t>(variant)));
  } else if (index < std::variant_size_v<variant_t>) {
    result = fast_visit<visitor_t, variant_t, index+1>(
      std::forward<visitor_t>(visitor),
      std::forward<variant_t>(variant)
    );
  } else {
      __builtin_unreachable();
  }
  return result;
}
}  // namespace raft
