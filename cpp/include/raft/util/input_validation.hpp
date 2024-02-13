/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

namespace raft {

template <class ElementType, class Extents, class Layout, class Accessor>
constexpr bool is_row_or_column_major(mdspan<ElementType, Extents, Layout, Accessor> m)
{
  return false;
}

template <class ElementType, class Extents, class Accessor>
constexpr bool is_row_or_column_major(mdspan<ElementType, Extents, layout_left, Accessor> m)
{
  return true;
}

template <class ElementType, class Extents, class Accessor>
constexpr bool is_row_or_column_major(mdspan<ElementType, Extents, layout_right, Accessor> m)
{
  return true;
}

template <class ElementType, class Extents, class Accessor>
constexpr bool is_row_or_column_major(mdspan<ElementType, Extents, layout_stride, Accessor> m)
{
  return is_row_major(m) || is_col_major(m);
}

template <class ElementType, class Extents, class Layout, class Accessor>
constexpr bool is_row_major(mdspan<ElementType, Extents, Layout, Accessor> /* m */)
{
  return false;
}

template <class ElementType, class Extents, class Accessor>
constexpr bool is_row_major(mdspan<ElementType, Extents, layout_left, Accessor> /* m */)
{
  return false;
}

template <class ElementType, class Extents, class Accessor>
constexpr bool is_row_major(mdspan<ElementType, Extents, layout_right, Accessor> /* m */)
{
  return true;
}

template <class ElementType, class Extents, class Accessor>
constexpr bool is_row_major(mdspan<ElementType, Extents, layout_stride, Accessor> m)
{
  return m.stride(1) == typename Extents::index_type(1) && m.stride(0) >= m.extent(1);
}

template <class ElementType, class Extents, class Layout, class Accessor>
constexpr bool is_col_major(mdspan<ElementType, Extents, Layout, Accessor> /* m */)
{
  return false;
}

template <class ElementType, class Extents, class Accessor>
constexpr bool is_col_major(mdspan<ElementType, Extents, layout_left, Accessor> /* m */)
{
  return true;
}

template <class ElementType, class Extents, class Accessor>
constexpr bool is_col_major(mdspan<ElementType, Extents, layout_right, Accessor> /* m */)
{
  return false;
}

template <class ElementType, class Extents, class Accessor>
constexpr bool is_col_major(mdspan<ElementType, Extents, layout_stride, Accessor> m)
{
  return m.stride(0) == typename Extents::index_type(1) && m.stride(1) >= m.extent(0);
}

template <class ElementType, class IndexType, size_t... Exts, class Layout, class Accessor>
constexpr bool is_matrix_view(
  mdspan<ElementType, extents<IndexType, Exts...>, Layout, Accessor> /* m */)
{
  return sizeof...(Exts) == 2;
}

template <class ElementType, class Extents>
constexpr bool is_matrix_view(mdspan<ElementType, Extents> m)
{
  return false;
}

template <class ElementType, class IndexType, size_t... Exts, class Layout, class Accessor>
constexpr bool is_vector_view(
  mdspan<ElementType, extents<IndexType, Exts...>, Layout, Accessor> /* m */)
{
  return sizeof...(Exts) == 1;
}

template <class ElementType, class Extents>
constexpr bool is_vector_view(mdspan<ElementType, Extents> m)
{
  return false;
}

template <class ElementType, class IndexType, size_t... Exts, class Layout, class Accessor>
constexpr bool is_scalar_view(
  mdspan<ElementType, extents<IndexType, Exts...>, Layout, Accessor> /* m */)
{
  return sizeof...(Exts) == 0;
}

template <class ElementType, class Extents>
constexpr bool is_scalar_view(mdspan<ElementType, Extents> m)
{
  return false;
}

};  // end namespace raft