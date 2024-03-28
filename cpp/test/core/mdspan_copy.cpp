/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "../test_utils.h"

#include <raft/core/copy.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#ifndef RAFT_DISABLE_CUDA
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#endif
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resources.hpp>

namespace raft {
TEST(MDSpanCopy, Mdspan1DHostHost)
{
  auto res     = resources{};
  auto cols    = std::uint32_t{2};
  auto in_left = make_host_vector<float, std::uint32_t, layout_c_contiguous>(res, cols);

  auto gen_unique_entry = [](auto&& x) { return x; };
  for (auto i = std::uint32_t{}; i < cols; ++i) {
    in_left(i) = gen_unique_entry(i);
  }

  auto out_right = make_host_vector<double, std::uint32_t, layout_f_contiguous>(res, cols);
  static_assert(detail::mdspan_copyable<true,
                                        decltype(out_right.view()),
                                        decltype(in_left.view())>::can_use_std_copy,
                "Current implementation should use std::copy for this copy");
  copy(res, out_right.view(), in_left.view());
  for (auto i = std::uint32_t{}; i < cols; ++i) {
    ASSERT_TRUE(match(out_right(i), double(gen_unique_entry(i)), CompareApprox<double>{0.0001}));
  }
}

#ifndef RAFT_DISABLE_CUDA
TEST(MDSpanCopy, Mdspan1DHostDevice)
{
  auto res     = device_resources{};
  auto cols    = std::uint32_t{2};
  auto in_left = make_host_vector<float, std::uint32_t, layout_c_contiguous>(res, cols);

  auto gen_unique_entry = [](auto&& x) { return x; };
  for (auto i = std::uint32_t{}; i < cols; ++i) {
    in_left(i) = gen_unique_entry(i);
  }

  auto out_right = make_device_vector<float, std::uint32_t, layout_f_contiguous>(res, cols);
  static_assert(detail::mdspan_copyable<true,
                                        decltype(out_right.view()),
                                        decltype(in_left.view())>::can_use_raft_copy,
                "Current implementation should use raft::copy for this copy");
  copy(res, out_right.view(), in_left.view());
  res.sync_stream();
  for (auto i = std::uint32_t{}; i < cols; ++i) {
    ASSERT_TRUE(
      match(float(out_right(i)), float(gen_unique_entry(i)), CompareApprox<float>{0.0001f}));
  }
}

TEST(MDSpanCopy, Mdspan1DDeviceHost)
{
  auto res     = device_resources{};
  auto cols    = std::uint32_t{2};
  auto in_left = make_device_vector<float, std::uint32_t, layout_c_contiguous>(res, cols);

  auto gen_unique_entry = [](auto&& x) { return x; };
  for (auto i = std::uint32_t{}; i < cols; ++i) {
    in_left(i) = gen_unique_entry(i);
  }

  auto out_right = make_host_vector<float, std::uint32_t, layout_f_contiguous>(res, cols);
  static_assert(detail::mdspan_copyable<true,
                                        decltype(out_right.view()),
                                        decltype(in_left.view())>::can_use_raft_copy,
                "Current implementation should use raft::copy for this copy");
  copy(res, out_right.view(), in_left.view());
  res.sync_stream();
  for (auto i = std::uint32_t{}; i < cols; ++i) {
    ASSERT_TRUE(
      match(float(out_right(i)), float(gen_unique_entry(i)), CompareApprox<float>{0.0001f}));
  }
}
#endif

TEST(MDSpanCopy, Mdspan3DHostHost)
{
  auto res             = resources{};
  auto constexpr depth = std::uint32_t{500};
  auto constexpr rows  = std::uint32_t{300};
  auto constexpr cols  = std::uint32_t{200};
  auto in_left = make_host_mdarray<float, std::uint32_t, layout_f_contiguous, depth, rows, cols>(
    res, extents<std::uint32_t, depth, rows, cols>{});
  auto in_right = make_host_mdarray<float, std::uint32_t, layout_c_contiguous, depth, rows, cols>(
    res, extents<std::uint32_t, depth, rows, cols>{});
  auto gen_unique_entry = [](auto&& x, auto&& y, auto&& z) { return x * 7 + y * 11 + z * 13; };

  for (auto i = std::uint32_t{}; i < depth; ++i) {
    for (auto j = std::uint32_t{}; j < rows; ++j) {
      for (auto k = std::uint32_t{}; k < cols; ++k) {
        in_left(i, j, k)  = gen_unique_entry(i, j, k);
        in_right(i, j, k) = gen_unique_entry(i, j, k);
      }
    }
  }

  auto out_left = make_host_mdarray<double, std::uint32_t, layout_f_contiguous, depth, rows, cols>(
    res, extents<std::uint32_t, depth, rows, cols>{});
  auto out_right = make_host_mdarray<double, std::uint32_t, layout_c_contiguous, depth, rows, cols>(
    res, extents<std::uint32_t, depth, rows, cols>{});

  static_assert(detail::mdspan_copyable<true,
                                        decltype(out_right.view()),
                                        decltype(in_right.view())>::can_use_std_copy,
                "Current implementation should use std::copy for this copy");
  copy(res, out_right.view(), in_right.view());
  for (auto i = std::uint32_t{}; i < depth; ++i) {
    for (auto j = std::uint32_t{}; j < rows; ++j) {
      for (auto k = std::uint32_t{}; k < cols; ++k) {
        ASSERT_TRUE(match(
          out_right(i, j, k), double(gen_unique_entry(i, j, k)), CompareApprox<double>{0.0001}));
      }
    }
  }

  copy(res, out_right.view(), in_left.view());
  for (auto i = std::uint32_t{}; i < depth; ++i) {
    for (auto j = std::uint32_t{}; j < rows; ++j) {
      for (auto k = std::uint32_t{}; k < cols; ++k) {
        ASSERT_TRUE(match(
          out_right(i, j, k), double(gen_unique_entry(i, j, k)), CompareApprox<double>{0.0001}));
      }
    }
  }

  copy(res, out_left.view(), in_right.view());
  for (auto i = std::uint32_t{}; i < depth; ++i) {
    for (auto j = std::uint32_t{}; j < rows; ++j) {
      for (auto k = std::uint32_t{}; k < cols; ++k) {
        ASSERT_TRUE(match(
          out_left(i, j, k), double(gen_unique_entry(i, j, k)), CompareApprox<double>{0.0001}));
      }
    }
  }

  static_assert(detail::mdspan_copyable<true, decltype(out_left.view()), decltype(in_left.view())>::
                  can_use_std_copy,
                "Current implementation should use std::copy for this copy");
  copy(res, out_left.view(), in_left.view());
  for (auto i = std::uint32_t{}; i < depth; ++i) {
    for (auto j = std::uint32_t{}; j < rows; ++j) {
      for (auto k = std::uint32_t{}; k < cols; ++k) {
        ASSERT_TRUE(match(
          out_left(i, j, k), double(gen_unique_entry(i, j, k)), CompareApprox<double>{0.0001}));
      }
    }
  }
}

#ifndef RAFT_DISABLE_CUDA
TEST(MDSpanCopy, Mdspan3DHostDevice)
{
  auto res = device_resources{};
  // Use smaller values here since host/device copy takes awhile.
  // Non-trivial logic is tested in the other cases.
  auto constexpr depth = std::uint32_t{5};
  auto constexpr rows  = std::uint32_t{3};
  auto constexpr cols  = std::uint32_t{2};
  auto in_left = make_host_mdarray<float, std::uint32_t, layout_c_contiguous, depth, rows, cols>(
    res, extents<std::uint32_t, depth, rows, cols>{});
  auto in_right = make_host_mdarray<float, std::uint32_t, layout_f_contiguous, depth, rows, cols>(
    res, extents<std::uint32_t, depth, rows, cols>{});
  auto gen_unique_entry = [](auto&& x, auto&& y, auto&& z) { return x * 7 + y * 11 + z * 13; };

  for (auto i = std::uint32_t{}; i < depth; ++i) {
    for (auto j = std::uint32_t{}; j < rows; ++j) {
      for (auto k = std::uint32_t{}; k < cols; ++k) {
        in_left(i, j, k)  = gen_unique_entry(i, j, k);
        in_right(i, j, k) = gen_unique_entry(i, j, k);
      }
    }
  }

  auto out_left = make_device_mdarray<float, std::uint32_t, layout_c_contiguous, depth, rows, cols>(
    res, extents<std::uint32_t, depth, rows, cols>{});
  auto out_right =
    make_device_mdarray<float, std::uint32_t, layout_f_contiguous, depth, rows, cols>(
      res, extents<std::uint32_t, depth, rows, cols>{});

  static_assert(detail::mdspan_copyable<true,
                                        decltype(out_right.view()),
                                        decltype(in_right.view())>::can_use_raft_copy,
                "Current implementation should use raft::copy for this copy");
  copy(res, out_right.view(), in_right.view());
  res.sync_stream();
  for (auto i = std::uint32_t{}; i < depth; ++i) {
    for (auto j = std::uint32_t{}; j < rows; ++j) {
      for (auto k = std::uint32_t{}; k < cols; ++k) {
        ASSERT_TRUE(match(float(out_right(i, j, k)),
                          float(gen_unique_entry(i, j, k)),
                          CompareApprox<float>{0.0001}));
      }
    }
  }

  static_assert(detail::mdspan_copyable<true, decltype(out_left.view()), decltype(in_left.view())>::
                  can_use_raft_copy,
                "Current implementation should use raft::copy for this copy");
  copy(res, out_left.view(), in_left.view());
  res.sync_stream();
  for (auto i = std::uint32_t{}; i < depth; ++i) {
    for (auto j = std::uint32_t{}; j < rows; ++j) {
      for (auto k = std::uint32_t{}; k < cols; ++k) {
        ASSERT_TRUE(match(float(out_left(i, j, k)),
                          float(gen_unique_entry(i, j, k)),
                          CompareApprox<float>{0.0001}));
      }
    }
  }
}

TEST(MDSpanCopy, Mdspan2DDeviceDevice)
{
  auto res            = device_resources{};
  auto constexpr rows = std::uint32_t{300};
  auto constexpr cols = std::uint32_t{200};
  auto in_left        = make_device_mdarray<float, std::uint32_t, layout_c_contiguous, rows, cols>(
    res, extents<std::uint32_t, rows, cols>{});
  auto in_right = make_device_mdarray<float, std::uint32_t, layout_f_contiguous, rows, cols>(
    res, extents<std::uint32_t, rows, cols>{});
  auto gen_unique_entry = [](auto&& x, auto&& y) { return x * 7 + y * 11; };

  for (auto i = std::uint32_t{}; i < rows; ++i) {
    for (auto j = std::uint32_t{}; j < cols; ++j) {
      in_left(i, j)  = gen_unique_entry(i, j);
      in_right(i, j) = gen_unique_entry(i, j);
    }
  }

  auto out_left = make_device_mdarray<float, std::uint32_t, layout_c_contiguous, rows, cols>(
    res, extents<std::uint32_t, rows, cols>{});
  auto out_right = make_device_mdarray<float, std::uint32_t, layout_f_contiguous, rows, cols>(
    res, extents<std::uint32_t, rows, cols>{});

  static_assert(detail::mdspan_copyable<true,
                                        decltype(out_right.view()),
                                        decltype(in_right.view())>::can_use_raft_copy,
                "Current implementation should use raft::copy for this copy");
  copy(res, out_right.view(), in_right.view());
  res.sync_stream();
  for (auto i = std::uint32_t{}; i < rows; ++i) {
    for (auto j = std::uint32_t{}; j < cols; ++j) {
      ASSERT_TRUE(
        match(float(out_right(i, j)), float(gen_unique_entry(i, j)), CompareApprox<float>{0.0001}));
    }
  }

  static_assert(detail::mdspan_copyable<true,
                                        decltype(out_right.view()),
                                        decltype(in_left.view())>::can_use_cublas,
                "Current implementation should use cuBLAS for this copy");
  copy(res, out_right.view(), in_left.view());
  res.sync_stream();
  for (auto i = std::uint32_t{}; i < rows; ++i) {
    for (auto j = std::uint32_t{}; j < cols; ++j) {
      ASSERT_TRUE(
        match(float(out_right(i, j)), float(gen_unique_entry(i, j)), CompareApprox<float>{0.0001}));
    }
  }

  static_assert(detail::mdspan_copyable<true,
                                        decltype(out_left.view()),
                                        decltype(in_right.view())>::can_use_cublas,
                "Current implementation should use cuBLAS for this copy");
  copy(res, out_left.view(), in_right.view());
  res.sync_stream();
  for (auto i = std::uint32_t{}; i < rows; ++i) {
    for (auto j = std::uint32_t{}; j < cols; ++j) {
      ASSERT_TRUE(
        match(float(out_left(i, j)), float(gen_unique_entry(i, j)), CompareApprox<float>{0.0001}));
    }
  }
}
#endif

}  // namespace raft
