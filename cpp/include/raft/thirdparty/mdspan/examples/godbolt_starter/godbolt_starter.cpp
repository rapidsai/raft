#if defined(_MDSPAN_USE_CLASS_TEMPLATE_ARGUMENT_DEDUCTION)
// https://godbolt.org/z/ehErvsTce
#include <experimental/mdspan>
#include <iostream>

namespace stdex = std::experimental;

int main() {
  std::array d{
    0, 5, 1,
    3, 8, 4,
    2, 7, 6,
  };

  stdex::mdspan m{d.data(), stdex::extents{3, 3}};

  for (std::size_t i = 0; i < m.extent(0); ++i)
    for (std::size_t j = 0; j < m.extent(1); ++j)
      std::cout << "m(" << i << ", " << j << ") == " << m(i, j) << "\n";
}
#else
int main() {}
#endif
