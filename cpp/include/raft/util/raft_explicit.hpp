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

#define RAFT_EXPLICIT                                                     \
  {                                                                       \
    raft::util::raft_explicit::do_not_implicitly_instantiate_templates(); \
  }

namespace raft::util::raft_explicit {

// To make sure the static_assert only fires when
// do_not_implicitly_instantiate_templates is instantiated, we use a dummy
// template parameter as described in P2593:
// https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2022/p2593r0.html
template <bool implicit_instantiation_allowed = false>
void do_not_implicitly_instantiate_templates()
{
  static_assert(implicit_instantiation_allowed,
                "ACCIDENTAL_IMPLICIT_INSTANTIATION\n\n"

                "If you see this error, then you have implicitly instantiated a function\n"
                "template. To keep compile times in check, libfoo has the policy of\n"
                "explicitly instantiating templates. To fix the compilation error, follow\n"
                "these steps.\n\n"

                "If you scroll up a bit in your error message, you probably saw two lines\n"
                "like the following:\n\n"

                "[.. snip ..] required from ‘void raft::do_not_implicitly_instantiate_templates() "
                "[with int dummy = 0]’\n"
                "[.. snip ..] from ‘void raft::bar(T) [with T = double]’\n\n"

                "Simple solution:\n\n"

                "    Add '#undef RAFT_EXPLICIT_INSTANTIATE' at the top of your .cpp/.cu file.\n\n"

                "Best solution:\n\n"

                "    1. Add the following line to the file include/raft/bar.hpp:\n\n"

                "        extern template void raft::bar<double>(double);\n\n"

                "    2. Add the following line to the file src/raft/bar.cpp:\n\n"

                "        template void raft::bar<double>(double)\n\n"

                "Probability is that there are many other similar lines in both files.\n");
}

}  // namespace raft::util::raft_explicit
