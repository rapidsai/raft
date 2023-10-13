/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <algorithm>
#include <execution>
#include <optional>
#include <raft/core/cuda_support.hpp>
#include <raft/core/detail/copy.hpp>
#include <raft/core/device_container_policy.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_container_policy.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/stream_view.hpp>
#include <raft/util/variant_utils.hpp>
#include <utility>
#include <variant>
#ifndef RAFT_DISABLE_CUDA
#include <raft/core/copy.cuh>
#include <raft/util/cudart_utils.hpp>
#include <thrust/device_ptr.h>
#else
#include <raft/core/copy.hpp>
#endif

namespace raft {

inline auto constexpr variant_index_from_memory_type(raft::memory_type mem_type)
{
  return static_cast<std::underlying_type_t<raft::memory_type>>(mem_type);
}

template <raft::memory_type MemType, typename Variant>
using alternate_from_mem_type =
  std::variant_alternative_t<variant_index_from_memory_type(MemType) % std::variant_size_v<Variant>,
                             Variant>;

template <typename T>
using default_container_policy_variant = std::variant<host_vector_policy<T>,
                                                      device_uvector_policy<T>,
                                                      managed_uvector_policy<T>,
                                                      pinned_vector_policy<T>>;

template <typename ElementType,
          typename ContainerPolicyVariant = default_container_policy_variant<ElementType>>
struct default_buffer_container_policy {
  using element_type = ElementType;
  using value_type   = std::remove_cv_t<element_type>;

  using container_policy_variant = ContainerPolicyVariant;

  template <raft::memory_type MemType>
  using container_policy =
    host_device_accessor<alternate_from_mem_type<MemType, container_policy_variant>, MemType>;

 private:
  template <std::size_t index>
  using container_policy_at_index = std::variant_alternative_t<index, container_policy_variant>;

 public:
  using container_type_variant =
    std::variant<typename container_policy_at_index<0>::container_type,
                 typename container_policy_at_index<1>::container_type,
                 typename container_policy_at_index<2>::container_type,
                 typename container_policy_at_index<3>::container_type>;

  template <raft::memory_type MemType>
  using container_type = alternate_from_mem_type<MemType, container_type_variant>;

  using accessor_policy_variant =
    std::variant<typename container_policy_at_index<0>::accessor_policy,
                 typename container_policy_at_index<1>::accessor_policy,
                 typename container_policy_at_index<2>::accessor_policy,
                 typename container_policy_at_index<3>::accessor_policy>;

  template <raft::memory_type MemType>
  using accessor_policy = alternate_from_mem_type<MemType, accessor_policy_variant>;

  using const_accessor_policy_variant =
    std::variant<typename container_policy_at_index<0>::const_accessor_policy,
                 typename container_policy_at_index<1>::const_accessor_policy,
                 typename container_policy_at_index<2>::const_accessor_policy,
                 typename container_policy_at_index<3>::const_accessor_policy>;

  template <raft::memory_type MemType>
  using const_accessor_policy = alternate_from_mem_type<MemType, accessor_policy_variant>;

  template <raft::memory_type MemType>
  auto create(raft::resources const& res, size_t n)
  {
    return container_type<MemType>(res, n);
  }

  auto create(raft::resources const& res, size_t n, raft::memory_type mem_type)
  {
    auto result = container_type_variant{};
    switch (mem_type) {
      case raft::memory_type::host: result = create<raft::memory_type::host>(res, n); break;
      case raft::memory_type::device: result = create<raft::memory_type::device>(res, n); break;
      case raft::memory_type::managed: result = create<raft::memory_type::managed>(res, n); break;
      case raft::memory_type::pinned: result = create<raft::memory_type::pinned>(res, n); break;
    }
    return result;
  }

 private:
  template <typename ContainerType>
  auto static constexpr has_stream() -> decltype(std::declval<ContainerType>().stream(), bool())
  {
    return true;
  };
  auto static constexpr has_stream(...) -> bool { return false; };

 public:
  template <memory_type MemType>
  [[nodiscard]] auto make_accessor_policy() noexcept
  {
    return accessor_policy<MemType>{};
  }
  template <memory_type MemType>
  [[nodiscard]] auto make_accessor_policy() const noexcept
  {
    return const_accessor_policy<MemType>{};
  }

  [[nodiscard]] auto make_accessor_policy(memory_type mem_type) noexcept
  {
    auto result = accessor_policy_variant{};
    switch (mem_type) {
      case memory_type::host: result = make_accessor_policy<memory_type::host>(); break;
      case memory_type::device: result = make_accessor_policy<memory_type::device>(); break;
      case memory_type::managed: result = make_accessor_policy<memory_type::managed>(); break;
      case memory_type::pinned: result = make_accessor_policy<memory_type::pinned>(); break;
    }
    return result;
  }
  [[nodiscard]] auto make_accessor_policy(memory_type mem_type) const noexcept
  {
    auto result = const_accessor_policy_variant{};
    switch (mem_type) {
      case memory_type::host: result = make_accessor_policy<memory_type::host>(); break;
      case memory_type::device: result = make_accessor_policy<memory_type::device>(); break;
      case memory_type::managed: result = make_accessor_policy<memory_type::managed>(); break;
      case memory_type::pinned: result = make_accessor_policy<memory_type::pinned>(); break;
    }
    return result;
  }
};

template <bool B, typename VariantT>
struct is_variant_of_mdspans : std::false_type {};

template <typename... AllVariantTs>
struct is_variant_of_mdspans<true, std::variant<AllVariantTs...>>
  : std::conjunction<is_mdspan<AllVariantTs>...> {};

template <typename VariantT>
auto static constexpr const is_variant_of_mdspans_v = is_variant_of_mdspans<true, VariantT>::value;

template <typename ElementType,
          typename Extents,
          typename LayoutPolicy    = layout_c_contiguous,
          typename ContainerPolicy = default_buffer_container_policy<ElementType>>
struct mdbuffer {
  using extents_type = Extents;
  using layout_type  = LayoutPolicy;
  using mapping_type = typename layout_type::template mapping<extents_type>;
  using element_type = ElementType;

  using value_type      = std::remove_cv_t<element_type>;
  using index_type      = typename extents_type::index_type;
  using difference_type = std::ptrdiff_t;
  using rank_type       = typename extents_type::rank_type;

  using container_policy_type   = ContainerPolicy;
  using accessor_policy_variant = typename ContainerPolicy::accessor_policy_variant;

  template <raft::memory_type MemType>
  using accessor_policy = alternate_from_mem_type<MemType, accessor_policy_variant>;

  using container_type_variant = typename container_policy_type::container_type_variant;

  template <raft::memory_type MemType>
  using container_type = typename container_policy_type::template container_type<MemType>;

  template <memory_type MemType>
  using owning_type = mdarray<element_type,
                              extents_type,
                              layout_type,
                              typename container_policy_type::template container_policy<MemType>>;
  // We use the static cast here to ensure that the memory types appear in the
  // order expected for retrieving the correct variant alternative based on
  // memory type. Even if the memory types are re-arranged in the enum and
  // assigned different values, the logic should remain correct.
  using owning_type_variant = std::variant<owning_type<static_cast<memory_type>(0)>,
                                           owning_type<static_cast<memory_type>(1)>,
                                           owning_type<static_cast<memory_type>(2)>,
                                           owning_type<static_cast<memory_type>(3)>>;

  template <memory_type MemType>
  using view_type = typename owning_type<MemType>::view_type;

  using view_type_variant = std::variant<view_type<static_cast<memory_type>(0)>,
                                         view_type<static_cast<memory_type>(1)>,
                                         view_type<static_cast<memory_type>(2)>,
                                         view_type<static_cast<memory_type>(3)>>;

  template <memory_type MemType>
  using const_view_type         = typename owning_type<MemType>::const_view_type;
  using const_view_type_variant = std::variant<const_view_type<static_cast<memory_type>(0)>,
                                               const_view_type<static_cast<memory_type>(1)>,
                                               const_view_type<static_cast<memory_type>(2)>,
                                               const_view_type<static_cast<memory_type>(3)>>;

  using storage_type_variant = concatenated_variant_t<view_type_variant, owning_type_variant>;

  template <memory_type MemType, bool is_owning>
  using storage_type =
    std::variant_alternative_t<std::size_t{is_owning} * std::variant_size_v<view_type_variant> +
                                 std::size_t{variant_index_from_memory_type(MemType)},
                               storage_type_variant>;

  template <bool B, typename FromT, typename T = void>
  struct constructible_from : std::false_type {};

  template <typename FromT, typename T>
  class constructible_from<true, FromT, T> {
    template <typename U>
    auto static constexpr has_mdspan_view() -> decltype(std::declval<U>().view(), bool())
    {
      return is_variant_of_mdspans_v<decltype(std::declval<U>().view())> ||
             raft::is_mdspan_v<decltype(std::declval<U>().view())>;
    };
    auto static constexpr has_mdspan_view(...) -> bool { return false; };

    template <typename U>
    auto static constexpr has_mem_type() -> decltype(std::declval<U>().mem_type(), bool())
    {
      return true;
    };
    auto static constexpr has_mem_type(...) -> bool { return false; };

    auto static constexpr const from_has_mdspan_view = has_mdspan_view<FromT>();

    using from_mdspan_type_variant = std::conditional_t<
      from_has_mdspan_view,
      std::conditional_t<is_variant_of_mdspans_v<decltype(std::declval<FromT>().view())>,
                         decltype(std::declval<FromT>().view()),
                         std::variant<decltype(std::declval<FromT>().view())>>,
      FromT>;

   public:
    template <memory_type MemType>
    using from_mdspan_type = alternate_from_mem_type<MemType, from_mdspan_type_variant>;

    auto static constexpr const default_mem_type_destination = []() {
      if constexpr (is_host_mdspan_v<from_mdspan_type<memory_type::managed>> &&
                    is_device_mdspan_v<from_mdspan_type<memory_type::managed>>) {
        return memory_type::managed;
      } else if constexpr (is_device_mdspan_v<from_mdspan_type<memory_type::device>>) {
        return memory_type::device;
      } else if constexpr (is_host_mdspan_v<from_mdspan_type<memory_type::host>>) {
        return memory_type::host;
      } else if (CUDA_ENABLED) {
        return memory_type::device;
      } else {
        return memory_type::host;
      }
    }();

    auto static get_mem_type_from_input(FromT&& from)
    {
      if constexpr (is_host_mdspan_v<from_mdspan_type<memory_type::managed>> &&
                    is_device_mdspan_v<from_mdspan_type<memory_type::managed>>) {
        return memory_type::managed;
      } else if constexpr (is_device_mdspan_v<from_mdspan_type<memory_type::device>>) {
        return memory_type::device;
      } else if constexpr (is_host_mdspan_v<from_mdspan_type<memory_type::host>>) {
        return memory_type::host;
      } else if (CUDA_ENABLED) {
        return memory_type::device;
      } else {
        return memory_type::host;
      }
    }

    template <memory_type DstMemType, memory_type SrcMemType>
    auto static constexpr const is_copyable_memory_combination =
      detail::mdspan_copyable_v<view_type<DstMemType>, from_mdspan_type<SrcMemType>>;

    template <memory_type SrcMemType>
    auto static constexpr const is_copyable_to_any_memory_type =
      is_copyable_memory_combination<memory_type::host, SrcMemType> ||
      is_copyable_memory_combination<memory_type::device, SrcMemType> ||
      is_copyable_memory_combination<memory_type::managed, SrcMemType> ||
      is_copyable_memory_combination<memory_type::pinned, SrcMemType>;

    auto static constexpr const value = is_copyable_to_any_memory_type<memory_type::host> ||
                                        is_copyable_to_any_memory_type<memory_type::device> ||
                                        is_copyable_to_any_memory_type<memory_type::managed> ||
                                        is_copyable_to_any_memory_type<memory_type::pinned>;

    using type = std::enable_if_t<value, T>;

    template <typename U                                                       = FromT,
              std::enable_if_t<std::conjunction_v<std::is_same<U, FromT>,
                                                  std::bool_constant<from_has_mdspan_view>,
                                                  std::bool_constant<value>>>* = nullptr>
    auto static constexpr get_mdspan(U&& from) -> from_mdspan_type_variant
    {
      return from.view();
    }

    template <
      typename U = FromT,
      std::enable_if_t<
        std::conjunction_v<std::is_same<U, FromT>, is_mdspan<U>, std::bool_constant<value>>>* =
        nullptr>
    auto static constexpr const get_mdspan(U&& from)
    {
      return std::forward<U>(from);
    }
  };

  template <typename FromT, typename T = void>
  using constructible_from_t = typename constructible_from<true, FromT, T>::type;
  template <typename FromT, typename T = void>
  auto static constexpr constructible_from_v = constructible_from<true, FromT, T>::value;

  template <typename FromT, typename T = void>
  using movable_from_t = std::enable_if_t<
    std::conjunction_v<std::bool_constant<constructible_from_v<FromT>>,
                       std::bool_constant<std::is_convertible_v<FromT, storage_type_variant>>>,
    T>;

  constexpr mdbuffer() = default;

 private:
  storage_type_variant data_{};

 public:
  template <typename FromT, movable_from_t<FromT>* = nullptr>
  mdbuffer(raft::resources const& res,
           FromT&& other,
           memory_type mem_type =
             constructible_from<true, FromT, memory_type>::default_mem_type_destination)
    : data_{std::move(other)}
  {
  }

  template <typename FromT, constructible_from_t<FromT>* = nullptr>
  mdbuffer(raft::resources const& res,
           FromT const& other,
           memory_type mem_type =
             constructible_from<true, FromT, memory_type>::default_mem_type_destination)
    : data_{[res, &other, mem_type]() {
        using config = constructible_from<true, FromT, void>;
        auto result  = owning_type_variant{};
        switch (mem_type) {
          case memory_type::host: {
            auto tmp_result = owning_type<memory_type::host>{};
            raft::copy(res, tmp_result.view(), config::get_mdspan(std::forward<FromT>(other)));
            result = std::move(tmp_result);
            break;
          }
          case memory_type::device: {
            auto tmp_result = owning_type<memory_type::device>{};
            raft::copy(res, tmp_result.view(), config::get_mdspan(std::forward<FromT>(other)));
            result = std::move(tmp_result);
            break;
          }
          case memory_type::managed: {
            auto tmp_result = owning_type<memory_type::managed>{};
            raft::copy(res, tmp_result.view(), config::get_mdspan(std::forward<FromT>(other)));
            result = std::move(tmp_result);
            break;
          }
          case memory_type::pinned: {
            auto tmp_result = owning_type<memory_type::pinned>{};
            raft::copy(res, tmp_result.view(), config::get_mdspan(std::forward<FromT>(other)));
            result = std::move(tmp_result);
            break;
          }
        }
        return result;
      }()}
  {
  }

  template <typename LayoutType = layout_type, typename T = element_type, typename... SizeTypes>
  explicit constexpr mdbuffer(T* ptr, SizeTypes... dynamic_extents)
    : data_{[ptr, dynamic_extents...]() {
        auto result = view_type_variant{};
        switch (memory_type_from_pointer(ptr)) {
          case memory_type::host:
            result = view_type_variant{view_type<memory_type::host>{ptr, dynamic_extents...}};
            break;
          case memory_type::device:
            result = view_type_variant{view_type<memory_type::device>{ptr, dynamic_extents...}};
            break;
          case memory_type::managed:
            result = view_type_variant{view_type<memory_type::managed>{ptr, dynamic_extents...}};
            break;
          case memory_type::pinned:
            result = view_type_variant{view_type<memory_type::pinned>{ptr, dynamic_extents...}};
            break;
        }
        return result;
      }()}
  {
  }

  /* template <typename T, typename... SizeTypes>
  explicit constexpr mdbuffer(T* ptr, SizeTypes... dynamic_extents)
    : mdbuffer<layout_c_contiguous, T, SizeTypes...>{ptr, dynamic_extents...}
  {
  } */

  [[nodiscard]] auto constexpr mem_type() const
  {
    return static_cast<memory_type>(data_.index() % std::variant_size_v<owning_type_variant>);
  };

  [[nodiscard]] auto constexpr is_owning() const
  {
    return data_.index() >= std::variant_size_v<view_type_variant>;
  };

 private:
  static auto constexpr get_view_from_data(view_type_variant const& data) { return data; }
  static auto constexpr get_view_from_data(const_view_type_variant const& data) { return data; }
  static auto constexpr get_view_from_data(owning_type_variant& data)
  {
    return view_type_variant{data.view()};
  }
  static auto constexpr get_view_from_data(owning_type_variant const& data)
  {
    return const_view_type_variant{data.view()};
  }

  template <typename MemTypeConstant>
  [[nodiscard]] auto view()
  {
    auto variant_view = fast_visit([](auto&& inner) { return get_view_from_data(inner); }, data_);
    if constexpr (MemTypeConstant::value.has_value()) {
      return std::get<variant_index_from_memory_type(MemTypeConstant::value.value())>(variant_view);
    } else {
      return variant_view;
    }
  }

  template <typename MemTypeConstant>
  [[nodiscard]] auto view() const
  {
    auto variant_view = fast_visit([](auto&& inner) { return get_view_from_data(inner); }, data_);
    if constexpr (MemTypeConstant::value.has_value()) {
      return std::get<variant_index_from_memory_type(MemTypeConstant::value.value())>(variant_view);
    } else {
      return variant_view;
    }
  }

 public:
  template <memory_type mem_type>
  [[nodiscard]] auto view()
  {
    return view<memory_type_constant<mem_type>>();
  }
  template <memory_type mem_type>
  [[nodiscard]] auto view() const
  {
    return view<memory_type_constant<mem_type>>();
  }
};

template <typename FromT>
mdbuffer(raft::resources const& res, FromT&& other, memory_type mem_type)
  -> mdbuffer<typename std::decay_t<FromT>::element_type,
              typename std::decay_t<FromT>::extents_type,
              typename std::decay_t<FromT>::layout_type>;

template <typename FromT>
mdbuffer(raft::resources const& res, FromT const& other, memory_type mem_type)
  -> mdbuffer<typename std::decay_t<FromT>::element_type,
              typename std::decay_t<FromT>::extents_type,
              typename std::decay_t<FromT>::layout_type>;

template <typename LayoutType, typename T, typename... SizeTypes>
mdbuffer(T* ptr, SizeTypes... dynamic_extents)
  -> mdbuffer<T, decltype(make_extents(dynamic_extents...)), LayoutType>;

}  // namespace raft
