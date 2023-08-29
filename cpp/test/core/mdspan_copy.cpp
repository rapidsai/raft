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

#include <cstdint>
#include <gtest/gtest.h>
#include <raft/core/mdspan_copy.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include "../test_utils.h"

namespace raft {
TEST(MDSpanCopy, Mdspan1D) {
  auto res = device_resources{};
  auto cols = std::uint32_t{2};
  auto in = make_host_vector<float>(res, cols);

  auto gen_unique_entry = [](auto&& x) {
    return x;
  };
  for (auto i=std::uint32_t{}; i < cols; ++i) {
    in(i) = gen_unique_entry(i);
  }

  auto out_different_contiguous_layout = make_host_vector<double, std::uint32_t, layout_f_contiguous>(res, cols);
  copy(res, out_different_contiguous_layout.view(), in.view());
  for (auto i=std::uint32_t{}; i < cols; ++i) {
    ASSERT_TRUE(match(out_different_contiguous_layout(i), double(gen_unique_entry(i)), CompareApprox<double>{0.0001}));
  }
}

TEST(MDSpanCopy, Mdspan3D) {
  auto res = device_resources{};
  auto constexpr depth = std::uint32_t{5};
  auto constexpr rows = std::uint32_t{3};
  auto constexpr cols = std::uint32_t{2};
  auto in = make_host_mdarray<float, std::uint32_t, layout_c_contiguous, depth, rows, cols>(
    res,
    extents<std::uint32_t, depth, rows, cols>{}
  );
  auto gen_unique_entry = [](auto&& x, auto&& y, auto&& z) {
    return x * 7 + y * 11 + z * 13;
  };

  for (auto i=std::uint32_t{}; i < depth; ++i) {
    for (auto j=std::uint32_t{}; j < rows; ++j) {
      for (auto k=std::uint32_t{}; k < cols; ++k) {
        in(i, j, k) = gen_unique_entry(i, j, k);
      }
    }
  }

  auto out_different_contiguous_layout = make_host_mdarray<double, std::uint32_t, layout_f_contiguous, depth, rows, cols>(
    res,
    extents<std::uint32_t, depth, rows, cols>{}
  );
  copy(res, out_different_contiguous_layout.view(), in.view());

  for (auto i=std::uint32_t{}; i < depth; ++i) {
    for (auto j=std::uint32_t{}; j < rows; ++j) {
      for (auto k=std::uint32_t{}; k < cols; ++k) {
        ASSERT_TRUE(match(
          out_different_contiguous_layout(i, j, k),
          double(gen_unique_entry(i, j, k)),
          CompareApprox<double>{0.0001}
        ));
      }
    }
  }

}
} // namespace raft
