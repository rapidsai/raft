/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <gtest/gtest.h>
#include <raft/core/mdarray.hpp>

namespace raft {

namespace stdex = std::experimental;

template <typename ElementType,
          typename Extents,
          typename LayoutPolicy   = layout_c_contiguous,
          typename AccessorPolicy = detail::stdex::default_accessor<ElementType>>
struct derived_device_mdspan
  : public device_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy> {
};

void test_template_asserts()
{
  // Testing 3d device mdspan to be an mdspan
  using three_d_extents = extents<int, dynamic_extent, dynamic_extent, dynamic_extent>;
  using three_d_mdspan  = device_mdspan<int, three_d_extents>;
  using d_mdspan        = derived_device_mdspan<int, three_d_extents>;

  static_assert(
    std::is_same_v<device_matrix_view<int, int>, device_mdspan<int, detail::matrix_extent<int>>>,
    "not same");
  static_assert(std::is_same_v<device_matrix_view<int, int>,
                               device_mdspan<int, extents<int, dynamic_extent, dynamic_extent>>>,
                "not same");

  // Checking if types are mdspan, supposed to fail for std::vector
  static_assert(is_mdspan_v<three_d_mdspan>, "3d mdspan type not an mdspan");
  static_assert(is_mdspan_v<device_matrix_view<float>>, "device_matrix_view type not an mdspan");
  static_assert(is_mdspan_v<const host_vector_view<unsigned long>>,
                "const host_vector_view type not an mdspan");
  static_assert(is_mdspan_v<const host_scalar_view<double>>,
                "const host_scalar_view type not an mdspan");
  static_assert(!is_mdspan_v<std::vector<int>>, "std::vector is an mdspan");
  static_assert(is_mdspan_v<d_mdspan>, "Derived device mdspan type is not mdspan");

  // Checking if types are device_mdspan
  static_assert(is_device_mdspan_v<device_matrix_view<float>>,
                "device_matrix_view type not a device_mdspan");
  static_assert(!is_device_mdspan_v<host_matrix_view<float>>,
                "host_matrix_view type is a device_mdspan");
  static_assert(is_device_mdspan_v<d_mdspan>, "Derived device mdspan type is not device_mdspan");

  // Checking if types are host_mdspan
  static_assert(!is_host_mdspan_v<device_matrix_view<float>>,
                "device_matrix_view type is a host_mdspan");
  static_assert(is_host_mdspan_v<host_matrix_view<float>>,
                "host_matrix_view type is not a host_mdspan");

  // checking variadics
  static_assert(!is_mdspan_v<three_d_mdspan, std::vector<int>>, "variadics mdspans");
  static_assert(is_mdspan_v<three_d_mdspan, d_mdspan>, "variadics not mdspans");
}

TEST(MDSpan, TemplateAsserts) { test_template_asserts(); }

void test_host_flatten()
{
  // flatten 3d host mdspan
  {
    using three_d_extents = extents<int, dynamic_extent, dynamic_extent, dynamic_extent>;
    using three_d_mdarray = host_mdarray<int, three_d_extents>;

    three_d_extents extents{3, 3, 3};
    typename three_d_mdarray::mapping_type layout{extents};
    typename three_d_mdarray::container_policy_type policy;
    three_d_mdarray mda{layout, policy};

    auto flat_view = flatten(mda);

    static_assert(std::is_same_v<typename three_d_mdarray::layout_type,
                                 typename decltype(flat_view)::layout_type>,
                  "layouts not the same");

    ASSERT_EQ(flat_view.extents().rank(), 1);
    ASSERT_EQ(flat_view.size(), mda.size());
  }

  // flatten host vector
  {
    auto hv        = make_host_vector<int>(27);
    auto flat_view = flatten(hv.view());

    ASSERT_EQ(hv.extents().rank(), flat_view.extents().rank());
    ASSERT_EQ(hv.extent(0), flat_view.extent(0));
  }

  // flatten host scalar
  {
    auto hs        = make_host_scalar<int>(27);
    auto flat_view = flatten(hs.view());

    ASSERT_EQ(flat_view.extent(0), 1);
  }
}

TEST(MDArray, HostFlatten) { test_host_flatten(); }

void test_device_flatten()
{
  raft::handle_t handle{};
  // flatten 3d device mdspan
  {
    using three_d_extents = extents<int, dynamic_extent, dynamic_extent, dynamic_extent>;
    using three_d_mdarray = device_mdarray<int, three_d_extents>;

    three_d_extents extents{3, 3, 3};
    typename three_d_mdarray::mapping_type layout{extents};
    typename three_d_mdarray::container_policy_type policy{handle.get_stream()};
    three_d_mdarray mda{layout, policy};

    auto flat_view = flatten(mda);

    static_assert(std::is_same_v<typename three_d_mdarray::layout_type,
                                 typename decltype(flat_view)::layout_type>,
                  "layouts not the same");

    ASSERT_EQ(flat_view.extents().rank(), 1);
    ASSERT_EQ(flat_view.size(), mda.size());
  }

  // flatten device vector
  {
    auto dv        = make_device_vector<int>(handle, 27);
    auto flat_view = flatten(dv.view());

    ASSERT_EQ(dv.extents().rank(), flat_view.extents().rank());
    ASSERT_EQ(dv.extent(0), flat_view.extent(0));
  }

  // flatten device scalar
  {
    auto ds        = make_device_scalar<int>(handle, 27);
    auto flat_view = flatten(ds.view());

    ASSERT_EQ(flat_view.extent(0), 1);
  }
}

TEST(MDArray, DeviceFlatten) { test_device_flatten(); }

void test_reshape()
{
  // reshape 3d host array to vector
  {
    using three_d_extents = extents<int, dynamic_extent, dynamic_extent, dynamic_extent>;
    using three_d_mdarray = host_mdarray<int, three_d_extents>;

    three_d_extents extents{3, 3, 3};
    typename three_d_mdarray::mapping_type layout{extents};
    typename three_d_mdarray::container_policy_type policy;
    three_d_mdarray mda{layout, policy};

    auto flat_view = reshape(mda, raft::extents<int, dynamic_extent>{27});

    ASSERT_EQ(flat_view.extents().rank(), 1);
    ASSERT_EQ(flat_view.size(), mda.size());
  }

  // reshape 4d device array to 2d
  {
    raft::handle_t handle{};
    using four_d_extents =
      extents<int, dynamic_extent, dynamic_extent, dynamic_extent, dynamic_extent>;
    using four_d_mdarray = device_mdarray<int, four_d_extents>;

    four_d_extents extents{2, 2, 2, 2};
    typename four_d_mdarray::mapping_type layout{extents};
    typename four_d_mdarray::container_policy_type policy{handle.get_stream()};
    four_d_mdarray mda{layout, policy};

    auto matrix = reshape(mda, raft::extents<int, dynamic_extent, dynamic_extent>{4, 4});

    ASSERT_EQ(matrix.extents().rank(), 2);
    ASSERT_EQ(matrix.extent(0), 4);
    ASSERT_EQ(matrix.extent(1), 4);
  }

  // reshape 2d host matrix with static extents to vector
  {
    using two_d_extents = extents<int, 5, 5>;
    using two_d_mdarray = host_mdarray<float, two_d_extents>;

    typename two_d_mdarray::mapping_type layout{two_d_extents{}};
    typename two_d_mdarray::container_policy_type policy;
    two_d_mdarray mda{layout, policy};

    auto vector = reshape(mda, extents<int, 25>{});

    ASSERT_EQ(vector.extents().rank(), 1);
    ASSERT_EQ(vector.size(), mda.size());
  }
}

TEST(MDArray, Reshape) { test_reshape(); }

}  // namespace raft