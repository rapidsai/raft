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
#include <raft/mdarray.hpp>

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
  using three_d_extents = stdex::extents<dynamic_extent, dynamic_extent, dynamic_extent>;
  using three_d_mdspan  = device_mdspan<int, three_d_extents>;
  using d_mdspan        = derived_device_mdspan<int, three_d_extents>;

  static_assert(std::is_same_v<device_matrix_view<int>, device_mdspan<int, detail::matrix_extent>>,
                "not same");
  static_assert(std::is_same_v<device_matrix_view<int>,
                               device_mdspan<int, stdex::extents<dynamic_extent, dynamic_extent>>>,
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
                "device_matrix_view type not a host_mdspan");
  static_assert(is_host_mdspan_v<host_matrix_view<float>>,
                "host_matrix_view type is a host_mdspan");
}

TEST(MDSpan, TemplateAsserts) { test_template_asserts(); }

void test_host_flatten()
{
  // flatten 3d host matrix
  {
    using three_d_extents = stdex::extents<dynamic_extent, dynamic_extent, dynamic_extent>;
    using three_d_mdarray = host_mdarray<int, three_d_extents>;

    three_d_extents extents{3, 3, 3};
    three_d_mdarray::container_policy_type policy;
    three_d_mdarray mda{extents, policy};

    auto flat_view = flatten(mda);

    static_assert(std::is_same_v<typename three_d_mdarray::layout_type,
                                 typename decltype(flat_view)::layout_type>,
                  "layouts not the same");

    ASSERT_EQ(flat_view.extents().rank(), 1);
    ASSERT_EQ(flat_view.extent(0), 27);
  }

  // flatten host vector
  {
    auto hv        = make_host_vector<int>(27);
    auto flat_view = flatten(hv.view());

    static_assert(std::is_same_v<decltype(hv.view()), decltype(flat_view)>, "types not the same");

    ASSERT_EQ(hv.extents().rank(), flat_view.extents().rank());
    ASSERT_EQ(hv.extent(0), flat_view.extent(0));
  }

  // flatten host scalar
  {
    auto hs        = make_host_scalar<int>(27);
    auto flat_view = flatten(hs.view());

    static_assert(std::is_same_v<decltype(hs.view()), decltype(flat_view)>, "types not the same");

    ASSERT_EQ(flat_view.extent(0), 1);
  }
}

TEST(MDArray, HostFlatten) { test_host_flatten(); }

}  // namespace raft