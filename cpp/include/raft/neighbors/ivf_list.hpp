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

#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>
#include <raft/neighbors/ivf_list_types.hpp>
#include <raft/util/integer_utils.hpp>

#include <thrust/fill.h>

#include <fstream>
#include <memory>
#include <type_traits>

namespace raft::neighbors::ivf {

/** The data for a single IVF list. */
template <template <typename, typename...> typename SpecT,
          typename SizeT,
          typename... SpecExtraArgs>
list<SpecT, SizeT, SpecExtraArgs...>::list(raft::resources const& res,
                                           const spec_type& spec,
                                           size_type n_rows)
  : size{n_rows}, data{res}, indices{res}
{
  auto capacity = round_up_safe<SizeT>(n_rows, spec.align_max);
  if (n_rows < spec.align_max) {
    capacity = bound_by_power_of_two<SizeT>(std::max<SizeT>(n_rows, spec.align_min));
    capacity = std::min<SizeT>(capacity, spec.align_max);
  }
  try {
    data    = make_device_mdarray<value_type>(res, spec.make_list_extents(capacity));
    indices = make_device_vector<index_type, SizeT>(res, capacity);
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
  thrust::fill_n(resource::get_thrust_policy(res),
                 indices.data_handle(),
                 indices.size(),
                 ivf::kInvalidRecord<index_type>);
}

/**
 * Resize a list by the given id, so that it can contain the given number of records;
 * copy the data if necessary.
 */
template <typename ListT>
void resize_list(raft::resources const& res,
                 std::shared_ptr<ListT>& orig_list,  // NOLINT
                 const typename ListT::spec_type& spec,
                 typename ListT::size_type new_used_size,
                 typename ListT::size_type old_used_size)
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
  auto new_list = std::make_shared<ListT>(res, spec, new_used_size);
  if (old_used_size > 0) {
    auto copied_data_extents = spec.make_list_extents(old_used_size);
    auto copied_view =
      make_mdspan<typename ListT::value_type, typename ListT::size_type, row_major, false, true>(
        new_list->data.data_handle(), copied_data_extents);
    copy(copied_view.data_handle(),
         orig_list->data.data_handle(),
         copied_view.size(),
         resource::get_cuda_stream(res));
    copy(new_list->indices.data_handle(),
         orig_list->indices.data_handle(),
         old_used_size,
         resource::get_cuda_stream(res));
  }
  // swap the shared pointer content with the new list
  new_list.swap(orig_list);
}

template <typename ListT>
auto serialize_list(const raft::resources& handle,
                    std::ostream& os,
                    const ListT& ld,
                    const typename ListT::spec_type& store_spec,
                    std::optional<typename ListT::size_type> size_override = std::nullopt)
  -> enable_if_valid_list_t<ListT>
{
  using size_type = typename ListT::size_type;
  auto size       = size_override.value_or(ld.size.load());
  serialize_scalar(handle, os, size);
  if (size == 0) { return; }

  auto data_extents = store_spec.make_list_extents(size);
  auto data_array =
    make_host_mdarray<typename ListT::value_type, size_type, row_major>(data_extents);
  auto inds_array = make_host_mdarray<typename ListT::index_type, size_type, row_major>(
    make_extents<size_type>(size));
  copy(data_array.data_handle(),
       ld.data.data_handle(),
       data_array.size(),
       resource::get_cuda_stream(handle));
  copy(inds_array.data_handle(),
       ld.indices.data_handle(),
       inds_array.size(),
       resource::get_cuda_stream(handle));
  resource::sync_stream(handle);
  serialize_mdspan(handle, os, data_array.view());
  serialize_mdspan(handle, os, inds_array.view());
}

template <typename ListT>
auto serialize_list(const raft::resources& handle,
                    std::ostream& os,
                    const std::shared_ptr<ListT>& ld,
                    const typename ListT::spec_type& store_spec,
                    std::optional<typename ListT::size_type> size_override = std::nullopt)
  -> enable_if_valid_list_t<ListT>
{
  if (ld) {
    return serialize_list<ListT>(handle, os, *ld, store_spec, size_override);
  } else {
    return serialize_scalar(handle, os, typename ListT::size_type{0});
  }
}

template <typename ListT>
auto deserialize_list(const raft::resources& handle,
                      std::istream& is,
                      std::shared_ptr<ListT>& ld,
                      const typename ListT::spec_type& store_spec,
                      const typename ListT::spec_type& device_spec) -> enable_if_valid_list_t<ListT>
{
  using size_type = typename ListT::size_type;
  auto size       = deserialize_scalar<size_type>(handle, is);
  if (size == 0) { return ld.reset(); }
  std::make_shared<ListT>(handle, device_spec, size).swap(ld);
  auto data_extents = store_spec.make_list_extents(size);
  auto data_array =
    make_host_mdarray<typename ListT::value_type, size_type, row_major>(data_extents);
  auto inds_array = make_host_mdarray<typename ListT::index_type, size_type, row_major>(
    make_extents<size_type>(size));
  deserialize_mdspan(handle, is, data_array.view());
  deserialize_mdspan(handle, is, inds_array.view());
  copy(ld->data.data_handle(),
       data_array.data_handle(),
       data_array.size(),
       resource::get_cuda_stream(handle));
  // NB: copying exactly 'size' indices to leave the rest 'kInvalidRecord' intact.
  copy(
    ld->indices.data_handle(), inds_array.data_handle(), size, resource::get_cuda_stream(handle));
  // Make sure the data is copied from host to device before the host arrays get out of the scope.
  resource::sync_stream(handle);
}

}  // namespace raft::neighbors::ivf
