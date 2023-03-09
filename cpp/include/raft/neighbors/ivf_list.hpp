/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <raft/neighbors/ivf_list_types.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/serialize.hpp>
#include <raft/util/integer_utils.hpp>

#include <thrust/fill.h>

#include <fstream>
#include <memory>
#include <type_traits>

namespace raft::neighbors::ivf {

/** The data for a single IVF list. */
template <template <typename> typename SpecT, typename IdxT, typename SizeT>
list<SpecT, IdxT, SizeT>::list(raft::device_resources const& res,
                               const SpecT<SizeT>& spec,
                               SizeT n_rows)
  : size{n_rows}
{
  auto capacity = round_up_safe<SizeT>(n_rows, spec.align_max);
  if (n_rows < spec.align_max) {
    capacity = bound_by_power_of_two<SizeT>(std::max<SizeT>(n_rows, spec.align_min));
    capacity = std::min<SizeT>(capacity, spec.align_max);
  }
  try {
    data    = make_device_mdarray<value_type>(res, spec.make_list_extents(capacity));
    indices = make_device_vector<IdxT, SizeT>(res, capacity);
  } catch (std::bad_alloc& e) {
    RAFT_FAIL(
      "ivf::list: failed to allocate a big enough list to hold all data "
      "(requested size: %zu records, selected capacity: %zu records). "
      "Allocator exception: %s",
      size_t(size),
      size_t(capacity),
      e.what());
  }
  // Fill the index buffer with a pre-defined marker for easier debugging
  thrust::fill_n(
    res.get_thrust_policy(), indices.data_handle(), indices.size(), ivf::kInvalidRecord<IdxT>);
}

/**
 * Resize a list by the given id, so that it can contain the given number of records;
 * copy the data if necessary.
 */
template <template <typename> typename SpecT, typename IdxT, typename SizeT>
void resize_list(raft::device_resources const& res,
                 std::shared_ptr<list<SpecT, IdxT, SizeT>>& orig_list,  // NOLINT
                 const SpecT<SizeT>& spec,
                 SizeT new_used_size,
                 SizeT old_used_size)
{
  bool skip_resize = false;
  if (orig_list) {
    if (new_used_size <= orig_list->indices.extent(0)) {
      auto shared_list_size = old_used_size;
      if (new_used_size <= old_used_size ||
          orig_list->size.compare_exchange_strong(shared_list_size, new_used_size)) {
        // We don't need to resize the list if:
        //  1. The list exists
        //  2. The new size fits in the list
        //  3. The list doesn't grow or no-one else has grown it yet
        skip_resize = true;
      }
    }
  } else {
    old_used_size = 0;
  }
  if (skip_resize) { return; }
  auto new_list = std::make_shared<list<SpecT, IdxT, SizeT>>(res, spec, new_used_size);
  if (old_used_size > 0) {
    auto copied_data_extents = SpecT<size_t>{spec}.make_list_extents(old_used_size);
    auto copied_view =
      make_mdspan<typename SpecT<SizeT>::value_type, size_t, row_major, false, true>(
        new_list->data.data_handle(), copied_data_extents);
    copy(copied_view.data_handle(),
         orig_list->data.data_handle(),
         copied_view.size(),
         res.get_stream());
    copy(new_list->indices.data_handle(),
         orig_list->indices.data_handle(),
         old_used_size,
         res.get_stream());
  }
  // swap the shared pointer content with the new list
  new_list.swap(orig_list);
}

template <template <typename> typename SpecT, typename IdxT, typename SizeT>
void serialize_list(const raft::device_resources& handle,
                    std::ostream& os,
                    const list<SpecT, IdxT, SizeT>& ld,
                    const SpecT<SizeT>& store_spec,
                    std::optional<SizeT> size_override = std::nullopt)
{
  auto size = size_override.value_or(ld.size.load());
  serialize_scalar(handle, os, size);
  if (size == 0) { return; }

  auto data_extents = store_spec.make_list_extents(size);
  auto data_array =
    make_host_mdarray<typename SpecT<SizeT>::value_type, SizeT, row_major>(data_extents);
  auto inds_array = make_host_mdarray<IdxT, SizeT, row_major>(make_extents<SizeT>(size));
  copy(data_array.data_handle(), ld.data.data_handle(), data_array.size(), handle.get_stream());
  copy(inds_array.data_handle(), ld.indices.data_handle(), inds_array.size(), handle.get_stream());
  handle.sync_stream();
  serialize_mdspan(handle, os, data_array.view());
  serialize_mdspan(handle, os, inds_array.view());
}

template <template <typename> typename SpecT, typename IdxT, typename SizeT>
void serialize_list(const raft::device_resources& handle,
                    std::ostream& os,
                    const std::shared_ptr<const list<SpecT, IdxT, SizeT>>& ld,
                    const SpecT<SizeT>& store_spec,
                    std::optional<SizeT> size_override = std::nullopt)
{
  if (ld) {
    return serialize_list(handle, os, *ld, store_spec, size_override);
  } else {
    return serialize_scalar(handle, os, SizeT{0});
  }
}

template <template <typename> typename SpecT, typename IdxT, typename SizeT>
void deserialize_list(const raft::device_resources& handle,
                      std::istream& is,
                      std::shared_ptr<list<SpecT, IdxT, SizeT>>& ld,
                      const SpecT<SizeT>& store_spec,
                      const SpecT<SizeT>& device_spec)
{
  auto size = deserialize_scalar<SizeT>(handle, is);
  if (size == 0) { return ld.reset(); }
  std::make_shared<list<SpecT, IdxT, SizeT>>(handle, device_spec, size).swap(ld);
  auto data_extents = store_spec.make_list_extents(size);
  auto data_array =
    make_host_mdarray<typename SpecT<SizeT>::value_type, SizeT, row_major>(data_extents);
  auto inds_array = make_host_mdarray<IdxT, SizeT, row_major>(make_extents<SizeT>(size));
  deserialize_mdspan(handle, is, data_array.view());
  deserialize_mdspan(handle, is, inds_array.view());
  copy(ld->data.data_handle(), data_array.data_handle(), data_array.size(), handle.get_stream());
  // NB: copying exactly 'size' indices to leave the rest 'kInvalidRecord' intact.
  copy(ld->indices.data_handle(), inds_array.data_handle(), size, handle.get_stream());
  // Make sure the data is copied from host to device before the host arrays get out of the scope.
  handle.sync_stream();
}

}  // namespace raft::neighbors::ivf
