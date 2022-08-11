/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <experimental/mdspan>

#include <iostream>
#include <cassert>

namespace stdex = std::experimental;

// Simple tiled layout.
// Hard-coded for 2D, column-major across tiles
// and row-major within each tile
struct SimpleTileLayout2D {
  template <class Extents>
  struct mapping {

    // for simplicity
    static_assert(Extents::rank() == 2, "SimpleTileLayout2D is hard-coded for 2D layout");

    // for convenience
    using size_type = typename Extents::size_type;

    // constructor
    mapping(
      Extents const& exts,
      size_type row_tile,
      size_type col_tile
    ) noexcept
      : extents_(exts),
        row_tile_size_(row_tile),
        col_tile_size_(col_tile)
    {
      // For simplicity, don't worry about negatives/zeros/etc.
      assert(row_tile > 0);
      assert(col_tile > 0);
      assert(exts.extent(0) > 0);
      assert(exts.extent(1) > 0);
    }

    mapping() noexcept = default;
    mapping(mapping const&) noexcept = default;
    mapping(mapping&&) noexcept = default;
    mapping& operator=(mapping const&) noexcept = default;
    mapping& operator=(mapping&&) noexcept = default;
    ~mapping() noexcept = default;

    //------------------------------------------------------------
    // Helper members (not part of the layout concept)

    constexpr size_type
    n_row_tiles() const noexcept {
      return extents_.extent(0) / row_tile_size_ + size_type((extents_.extent(0) % row_tile_size_) != 0);
    }

    constexpr size_type
    n_column_tiles() const noexcept {
      return extents_.extent(1) / col_tile_size_ + size_type((extents_.extent(1) % col_tile_size_) != 0);
    }

    constexpr size_type
    tile_size() const noexcept {
      return row_tile_size_ * col_tile_size_;
    }

    size_type
    tile_offset(size_type row, size_type col) const noexcept {
      // This could probably be more efficient, but for example purposes...
      auto col_tile = col / col_tile_size_;
      auto row_tile = row / row_tile_size_;
      // We're hard-coding this to *column-major* layout across tiles
      return (col_tile * n_row_tiles() + row_tile) * tile_size();
    }

    size_type
    offset_in_tile(size_type row, size_type col) const noexcept {
      auto t_row = row % row_tile_size_;
      auto t_col = col % col_tile_size_;
      // We're hard-coding this to *row-major* within tiles
      return t_row * col_tile_size_ + t_col;
    }

    //------------------------------------------------------------
    // Required members

    constexpr size_type
    operator()(size_type row, size_type col) const noexcept {
      return tile_offset(row, col) + offset_in_tile(row, col);
    }

    constexpr size_type
    required_span_size() const noexcept {
      return n_row_tiles() * n_column_tiles() * tile_size();
    }

    // Mapping is always unique
    static constexpr bool is_always_unique() noexcept { return true; }
    // There is not always a regular stride between elements in a given dimension
    static constexpr bool is_always_strided() noexcept { return false; }
    // only contiguous if extents_.extent(0) % column_tile_size == 0, so not always
    static constexpr bool is_always_contiguous() noexcept { return false; }

    static constexpr bool is_unique() noexcept { return true; }
    constexpr bool
    is_contiguous() const noexcept {
      // Only contiguous if extents fit exactly into tile sizes...
      return (extents_.extent(0) % row_tile_size_ == 0) && (extents_.extent(1) % col_tile_size_ == 0);
    }
    // There are some circumstances where this is strided, but we're not
    // concerned about that optimization, so we're allowed to just return false here
    constexpr bool is_strided() const noexcept { return false; }

   private:

    Extents extents_;
    size_type row_tile_size_;
    size_type col_tile_size_;

  };
};



int main() {
  //----------------------------------------
  // 5x2 example, tiled as 3x3
  constexpr int n_rows = 2;
  constexpr int n_cols = 5;
  int data_row_major[] = {
       1,   2,   3,  /*|*/  4,   5,  /* X | */
       6,   7,   8,  /*|*/  9,  10   /* X | */
     /*X,   X,   X,    |    X    X      X |
      *------------------------------------ */
  };
  // manually tiling the above, using -1 as filler
  static constexpr int X = -1;
  int data_tiled[] = {
    /* tile 0, 0 */  1,  2,  3,  6,  7,  8,  X,  X,  X,
    /* tile 0, 1 */  4,  5,  X,  9, 10,  X,  X,  X,  X,
  };
  //----------------------------------------
  // Just use dynamic extents for the purposes of demonstration
  using extents_type = stdex::dextents<size_t, 2>;
  using tiled_mdspan = stdex::mdspan<int, extents_type, SimpleTileLayout2D>;
  using tiled_layout_type = typename SimpleTileLayout2D::template mapping<extents_type>;
  using row_major_mdspan = stdex::mdspan<int, extents_type, stdex::layout_right>;
  //----------------------------------------
  auto tiled = tiled_mdspan(data_tiled, tiled_layout_type(extents_type(n_rows, n_cols), 3, 3));
  auto row_major = row_major_mdspan(data_row_major, n_rows, n_cols);
  //----------------------------------------
  // Check that we did things correctly
  int failures = 0;
  for (int irow = 0; irow < n_rows; ++irow) {
    for (int icol = 0; icol < n_cols; ++icol) {
      if(tiled(irow, icol) != row_major(irow, icol)) {
        std::cout << "Mismatch for entry " << irow << ", " << icol << ":" << std::endl;
        std::cout << "  tiled(" << irow << ", " << icol << ") = "
                  << tiled(irow, icol) << std::endl;
        std::cout << "  row_major(" << irow << ", " << icol << ") = "
                  << row_major(irow, icol) << std::endl;
        ++failures;
      }

    }
  }
  if(failures == 0) {
    std::cout << "Success! SimpleTiledLayout2D works as expected." << std::endl;
  }
}

