#include <gtest/gtest.h>
#include <raft/mdarray.hpp>

namespace raft {

void test_template_asserts() {
    using three_d_extents = stdex::extents<dynamic_extent, dynamic_extent, dynamic_extent>;
    using three_d_mdspan = device_mdspan<int, three_d_extents>;

    static_assert(is_device_mdspan<three_d_mdspan>::value, "Not a device_mdspan");
}

TEST(MDspan, TemplateAsserts) {
    test_template_asserts();
}

} // namespace raft