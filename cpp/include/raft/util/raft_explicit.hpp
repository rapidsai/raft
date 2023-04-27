/* Copyright (c) 2023, NVIDIA CORPORATION.
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

/**
 * @brief Prevents a function template from being implicitly instantiated
 *
 * This macro defines a function body that can be used for function template
 * definitions of functions that should not be implicitly instantiated.
 *
 * When the template is erroneously implicitly instantiated, it provides a
 * useful error message that tells the user how to avoid the implicit
 * instantiation.
 *
 * The error message is generated using a static assert. It is generally tricky
 * to have a static assert fire only when you want it, as documented in
 * P2593: https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2593r0.html
 *
 * We use the strategy from paragraph 1.3 here. We define a struct
 * `not_allowed`, whose type is dependent on the template parameters of the
 * enclosing function instance. We use this struct type to instantiate the
 * `implicit_instantiation` template class, whose value is always false. We pass
 * this value to static_assert. This way, the static assert only fires when the
 * template is instantiated, since `implicit_instantiation` cannot be
 * instantiated without all the types in the enclosing function template.
 */
#define RAFT_EXPLICIT                                                                          \
  {                                                                                            \
    /* Type of `not_allowed` depends on template parameters of enclosing function. */          \
    struct not_allowed {};                                                                     \
    static_assert(                                                                             \
      raft::util::raft_explicit::implicit_instantiation<not_allowed>::value,                   \
      "ACCIDENTAL_IMPLICIT_INSTANTIATION\n\n"                                                  \
                                                                                               \
      "If you see this error, then you have implicitly instantiated a function\n"              \
      "template. To keep compile times in check, libraft has the policy of\n"                  \
      "explicitly instantiating templates. To fix the compilation error, follow\n"             \
      "these steps.\n\n"                                                                       \
                                                                                               \
      "If you scroll up or down a bit, you probably saw a line like the following:\n\n"        \
                                                                                               \
      "detected during instantiation of \"void raft::foo(T) [with T=float]\" at line [..]\n\n" \
                                                                                               \
      "Simplest temporary solution:\n\n"                                                       \
                                                                                               \
      "    Add '#undef RAFT_EXPLICIT_INSTANTIATE_ONLY' at the top of your .cpp/.cu file.\n\n"  \
                                                                                               \
      "Best solution:\n\n"                                                                     \
                                                                                               \
      "    1. Add the following line to the file include/raft/foo.hpp:\n\n"                    \
                                                                                               \
      "        extern template void raft::foo<double>(double);\n\n"                            \
                                                                                               \
      "    2. Add the following line to the file src/raft/foo.cpp:\n\n"                        \
                                                                                               \
      "        template void raft::foo<double>(double)\n");                                    \
                                                                                               \
    /* Function may have non-void return type. */                                              \
    /* To prevent warnings/errors about missing returns, throw an exception. */                \
    throw "raft_explicit_error";                                                               \
  }

namespace raft::util::raft_explicit {
/**
 * @brief Template that is always false
 *
 * This template is from paragraph 1.3 of P2593:
 * https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2593r0.html
 *
 * The value of `value` is always false, but it depends on a template parameter.
 */
template <typename T>
struct implicit_instantiation {
  static constexpr bool value = false;
};
}  // namespace raft::util::raft_explicit
