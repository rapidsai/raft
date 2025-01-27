/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
#ifdef NVTX_ENABLED
#include <raft/core/detail/nvtx.hpp>

#include <gtest/gtest.h>
/**
 * tests for the functionality of generating next color based on string
 * entered in the NVTX Range marker wrappers
 */

namespace raft {

class NvtxNextColorTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    const std::string temp1 = "foo";
    const std::string temp2 = "bar";

    diff_string_diff_color = common::nvtx::detail::generate_next_color(temp1) !=
                             common::nvtx::detail::generate_next_color(temp2);
    same_string_same_color = common::nvtx::detail::generate_next_color(temp1) ==
                             common::nvtx::detail::generate_next_color(temp1);
  }
  void TearDown() {}
  bool diff_string_diff_color = false;
  bool same_string_same_color = false;
};

TEST_F(NvtxNextColorTest, generate_next_color)
{
  EXPECT_TRUE(diff_string_diff_color);
  EXPECT_TRUE(same_string_same_color);
}

}  // end namespace raft
#endif
