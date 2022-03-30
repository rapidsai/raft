#include <gtest/gtest.h>
#include <raft/mdarray.hpp>

namespace raft {

namespace stdex = std::experimental;

void test_template_asserts() {
    // Testing 3d device mdspan to be an mdspan
    using three_d_extents = stdex::extents<dynamic_extent, dynamic_extent, dynamic_extent>;
    using three_d_mdspan = device_mdspan<int, three_d_extents>;

    static_assert(is_mdspan_v<three_d_mdspan>, "3d mdspan type not an mdspan");
    static_assert(is_mdspan_v<device_matrix_view<float>>, "device_matrix_view type not an mdspan");
    static_assert(is_mdspan_v<const host_vector_view<unsigned long>>, "const host_vector_view type not an mdspan");
    static_assert(is_mdspan_v<const host_scalar_view<double>>, "const host_scalar_view type not an mdspan");

    // static_assert(is_mdspan<three_d_mdspan, device_mdspan>::value, "Not an mdspan");
}

TEST(MDspan, TemplateAsserts) {
    test_template_asserts();
}

} // namespace raft