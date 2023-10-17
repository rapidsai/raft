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

  using container_policy_variant =
    std::variant<host_device_accessor<std::variant_alternative_t<0, ContainerPolicyVariant>, static_cast<memory_type>(0)>, host_device_accessor<std::variant_alternative_t<1, ContainerPolicyVariant>, static_cast<memory_type>(1)>, host_device_accessor<std::variant_alternative_t<2, ContainerPolicyVariant>, static_cast<memory_type>(2)>, host_device_accessor<std::variant_alternative_t<3, ContainerPolicyVariant>, static_cast<memory_type>(3)>, >;
  template <raft::memory_type MemType>
  using container_policy = alternate_from_mem_type<MemType, container_policy_variant>;

 private:
  template <std::size_t index>
  using container_policy_at_index = std::variant_alternative_t<index, ContainerPolicyVariant>;

 public:
  using container_type_variant =
    std::variant<typename container_policy_at_index<0>::container_type,
                 typename container_policy_at_index<1>::container_type,
                 typename container_policy_at_index<2>::container_type,
                 typename container_policy_at_index<3>::container_type>;

  template <raft::memory_type MemType>
  using container_type = alternate_from_mem_type<MemType, container_type_variant>;

  using accessor_policy_variant =
    std::variant<host_device_accessor<typename container_policy_at_index<0>::accessor_policy,
                                      static_cast<memory_type>(0)>,
                 host_device_accessor<typename container_policy_at_index<1>::accessor_policy,
                                      static_cast<memory_type>(1)>,
                 host_device_accessor<typename container_policy_at_index<2>::accessor_policy,
                                      static_cast<memory_type>(2)>,
                 host_device_accessor<typename container_policy_at_index<3>::accessor_policy,
                                      static_cast<memory_type>(3)>>;

  template <raft::memory_type MemType>
  using accessor_policy = alternate_from_mem_type<MemType, accessor_policy_variant>;

  using const_accessor_policy_variant =
    std::
      variant<host_device_accessor<typename container_policy_at_index<0>::const_accessor_policy, static_cast<memory_type>(0)>, host_device_accessor<typename container_policy_at_index<1>::const_accessor_policy, static_cast<memory_type>(1)>, host_device_accessor<typename container_policy_at_index<2>::const_accessor_policy, static_cast<memory_type>(2)>, host_device_accessor<typename container_policy_at_index<3>::const_accessor_policy, static_cast<memory_type>(3)>, >;

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

  constexpr mdbuffer() = default;

 private:
  container_policy_type cp_{};
  storage_type_variant data_{};

  template <typename FromT, std::size_t FromIndex, std::size_t ToIndex>
  auto static constexpr is_copyable_combination()
  {
    return detail::mdspan_copyable_v<
      decltype(std::declval<std::variant_alternative_t<ToIndex, owning_type_variant>>().view()),
      std::variant_alternative_t<FromIndex, decltype(std::declval<FromT>().view())>>;
  }

  template <std::size_t FromIndex, typename FromT, std::size_t... Is>
  auto static constexpr get_copyable_combinations(std::index_sequence<Is...>)
  {
    return std::array{is_copyable_combination<FromT, FromIndex, Is>()...};
  }

  template <typename FromT, std::size_t... Is>
  auto static constexpr get_copyable_combinations(bool, std::index_sequence<Is...>)
  {
    return std::array{get_copyable_combinations<Is, FromT>(
      std::make_index_sequence<std::variant_size_v<owning_type_variant>>())...};
  }

  template <typename FromT>
  auto static constexpr get_copyable_combinations()
  {
    return get_copyable_combinations(
      true,
      std::make_index_sequence<std::variant_size_v<decltype(std::declval<FromT>().view())>>());
  }

  template <std::size_t FromIndex, typename FromT, std::size_t... Is>
  auto static constexpr is_copyable_from(std::index_sequence<Is...>)
  {
    return (... || get_copyable_combinations<FromT>()[FromIndex][Is]);
  }

  template <typename FromT, std::size_t... Is>
  auto static constexpr is_copyable_from(bool, std::index_sequence<Is...>)
  {
    return (... || is_copyable_from<Is, FromT>(
                     std::make_index_sequence<std::variant_size_v<owning_type_variant>>()));
  }

  template <typename FromT>
  auto static constexpr is_copyable_from()
  {
    return is_copyable_from<FromT>(
      true,
      std::make_index_sequence<std::variant_size_v<decltype(std::declval<FromT>().view())>>());
  }

  template <typename FromT>
  auto static is_copyable_from(FromT&& other, memory_type mem_type)
  {
    auto static copyable_combinations = get_copyable_combinations<FromT>();
    return copyable_combinations[variant_index_from_memory_type(other.mem_type())]
                                [variant_index_from_memory_type(mem_type)];
  }

  template <typename FromT>
  auto static copy_from(raft::resources const& res, FromT&& other, memory_type mem_type)
  {
    auto result = storage_type_variant{};
    switch (mem_type) {
      case memory_type::host: {
        result = std::visit(
          [&res](auto&& other_view) {
            auto tmp_result = owning_type<memory_type::host>{
              res,
              layout_type{other_view.extents()},
              typename container_policy_type::template container_policy<memory_type::host>{}};
            raft::copy(res, tmp_result.view(), other_view);
            return tmp_result;
          },
          other.view());
        break;
      }
      case memory_type::device: {
        result = std::visit(
          [&res](auto&& other_view) {
            auto tmp_result = owning_type<memory_type::device>{
              res,
              layout_type{other_view.extents()},
              typename container_policy_type::template container_policy<memory_type::device>{}};
            raft::copy(res, tmp_result.view(), other_view);
            return tmp_result;
          },
          other.view());
        break;
      }
      case memory_type::managed: {
        result = std::visit(
          [&res](auto&& other_view) {
            auto tmp_result = owning_type<memory_type::managed>{
              res,
              layout_type{other_view.extents()},
              typename container_policy_type::template container_policy<memory_type::managed>{}};
            raft::copy(res, tmp_result.view(), other_view);
            return tmp_result;
          },
          other.view());
        break;
      }
      case memory_type::pinned: {
        result = std::visit(
          [&res](auto&& other_view) {
            auto tmp_result = owning_type<memory_type::host>{
              res,
              layout_type{other_view.extents()},
              typename container_policy_type::template container_policy<memory_type::host>{}};
            raft::copy(res, tmp_result.view(), other_view);
            return tmp_result;
          },
          other.view());
        break;
      }
    }
    return result;
  }

 public:
  template <
    typename OtherAccessorPolicy,
    std::enable_if_t<is_type_in_variant_v<OtherAccessorPolicy, accessor_policy_variant>>* = nullptr>
  mdbuffer(mdspan<ElementType, Extents, LayoutPolicy, OtherAccessorPolicy> other) : data_{other}
  {
  }

  template <typename OtherContainerPolicy,
            std::enable_if_t<is_type_in_variant_v<
              host_device_accessor<typename OtherContainerPolicy::accessor_type,
                                   OtherContainerPolicy::mem_type>,
              typename container_policy_type::container_policy_variant>>* = nullptr>
  mdbuffer(mdarray<ElementType, Extents, LayoutPolicy, OtherContainerPolicy>&& other)
    : data_{std::move(other)}
  {
  }

  template <typename OtherContainerPolicy,
            std::enable_if_t<is_type_in_variant_v<
              host_device_accessor<typename OtherContainerPolicy::accessor_type,
                                   OtherContainerPolicy::mem_type>,
              typename container_policy_type::container_policy_variant>>* = nullptr>
  mdbuffer(mdarray<ElementType, Extents, LayoutPolicy, OtherContainerPolicy>& other)
    : mdbuffer{other.view()}
  {
  }

  mdbuffer(raft::resources const& res,
           mdbuffer<ElementType, Extents, LayoutPolicy, ContainerPolicy>&& other,
           std::optional<memory_type> specified_mem_type = std::nullopt)
    : data_{[&res, &other, specified_mem_type, this]() {
        auto other_mem_type = other.mem_type();
        auto mem_type       = specified_mem_type.value_or(other_mem_type);
        auto result         = storage_type_variant{};
        if (mem_type == other.mem_type()) {
          result = std::move(other.data_);
        } else if (!other.is_owning() && has_compatible_accessibility(other_mem_type, mem_type)) {
          switch (mem_type) {
            case (memory_type::host): {
              result = std::visit(
                [&result, this](auto&& other_view) {
                  return view_type<memory_type::host>{
                    other_view.data_handle(),
                    other_view.mapping(),
                    cp_.template make_accessor_policy<memory_type::host>()};
                },
                other.view());
              break;
            }
            case (memory_type::device): {
              result = std::visit(
                [&result, this](auto&& other_view) {
                  return view_type<memory_type::device>{
                    other_view.data_handle(),
                    other_view.mapping(),
                    cp_.template make_accessor_policy<memory_type::device>()};
                },
                other.view());
              break;
            }
            case (memory_type::managed): {
              result = std::visit(
                [&result, this](auto&& other_view) {
                  return view_type<memory_type::managed>{
                    other_view.data_handle(),
                    other_view.mapping(),
                    cp_.template make_accessor_policy<memory_type::managed>()};
                },
                other.view());
              break;
            }
            case (memory_type::pinned): {
              result = std::visit(
                [&result, this](auto&& other_view) {
                  return view_type<memory_type::pinned>{
                    other_view.data_handle(),
                    other_view.mapping(),
                    cp_.template make_accessor_policy<memory_type::pinned>()};
                },
                other.view());
              break;
            }
          }
        } else {
          result = copy_from(res, other, mem_type);
        }
        return result;
      }()}
  {
  }

  mdbuffer(raft::resources const& res,
           mdbuffer<ElementType, Extents, LayoutPolicy, ContainerPolicy> const& other,
           std::optional<memory_type> specified_mem_type = std::nullopt)
    : data_{[&res, &other, specified_mem_type, this]() {
        auto mem_type       = specified_mem_type.value_or(other.mem_type());
        auto result         = storage_type_variant{};
        auto other_mem_type = other.mem_type();
        if (mem_type == other_mem_type) {
          result = std::visit([&result](auto&& other_view) { return other_view; }, other.view());
        } else if (has_compatible_accessibility(other_mem_type, mem_type)) {
          switch (mem_type) {
            case (memory_type::host): {
              result = std::visit(
                [&result, this](auto&& other_view) {
                  return view_type<memory_type::host>{
                    other_view.data_handle(),
                    other_view.mapping(),
                    cp_.template make_accessor_policy<memory_type::host>()};
                },
                other.view());
              break;
            }
            case (memory_type::device): {
              result = std::visit(
                [&result, this](auto&& other_view) {
                  return view_type<memory_type::device>{
                    other_view.data_handle(),
                    other_view.mapping(),
                    cp_.template make_accessor_policy<memory_type::device>()};
                },
                other.view());
              break;
            }
            case (memory_type::managed): {
              result = std::visit(
                [&result, this](auto&& other_view) {
                  return view_type<memory_type::managed>{
                    other_view.data_handle(),
                    other_view.mapping(),
                    cp_.template make_accessor_policy<memory_type::managed>()};
                },
                other.view());
              break;
            }
            case (memory_type::pinned): {
              result = std::visit(
                [&result, this](auto&& other_view) {
                  return view_type<memory_type::pinned>{
                    other_view.data_handle(),
                    other_view.mapping(),
                    cp_.template make_accessor_policy<memory_type::pinned>()};
                },
                other.view());
              break;
            }
          }
        } else {
          result = copy_from(res, other, mem_type);
        }
      }()}
  {
  }

  template <
    typename OtherElementType,
    typename OtherExtents,
    typename OtherLayoutPolicy,
    typename OtherContainerPolicy,
    std::enable_if_t<is_copyable_from<
      mdbuffer<OtherElementType, OtherExtents, OtherLayoutPolicy, OtherContainerPolicy>>()>* =
      nullptr>
  mdbuffer(
    raft::resources const& res,
    mdbuffer<OtherElementType, OtherExtents, OtherLayoutPolicy, OtherContainerPolicy> const& other,
    std::optional<memory_type> specified_mem_type = std::nullopt)
    : data_{[&res, &other, specified_mem_type]() {
        auto mem_type = specified_mem_type.value_or(other.mem_type());
        // Note: We perform this check at runtime because it is possible for two
        // mdbuffers to have storage types which may be copied to each other for
        // some memory types but not for others. This is an unusual situation, but
        // we still need to guard against it.
        RAFT_EXPECTS(
          is_copyable_from<decltype(other)>(other, mem_type),
          "mdbuffer cannot be constructed from other mdbuffer with indicated memory type");
        copy_from(res, other, mem_type);
      }()}
  {
  }

  [[nodiscard]] auto constexpr mem_type() const
  {
    return static_cast<memory_type>(data_.index() % std::variant_size_v<owning_type_variant>);
  };

  [[nodiscard]] auto constexpr is_owning() const
  {
    return data_.index() >= std::variant_size_v<view_type_variant>;
  };

 private:
  template <typename MemTypeConstant>
  [[nodiscard]] auto view()
  {
    if constexpr (MemTypeConstant::value.has_value()) {
      if (is_owning()) {
        return std::get<owning_type<MemTypeConstant::value.value()>>(data_).view();
      } else {
        return std::get<view_type<MemTypeConstant::value.value()>>(data_);
      }
    } else {
      return std::visit(
        [](auto&& inner) {
          if constexpr (is_mdspan_v<std::remove_reference_t<decltype(inner)>>) {
            return view_type_variant{inner};
          } else {
            return view_type_variant{inner.view()};
          }
        },
        data_);
    }
  }

  template <typename MemTypeConstant>
  [[nodiscard]] auto view() const
  {
    if constexpr (MemTypeConstant::value.has_value()) {
      if (is_owning()) {
        return make_const_mdspan(
          std::get<owning_type<MemTypeConstant::value.value()>>(data_).view());
      } else {
        return make_const_mdspan(std::get<view_type<MemTypeConstant::value.value()>>(data_));
      }
    } else {
      return std::visit(
        [](auto&& inner) {
          if constexpr (is_mdspan_v<std::remove_reference_t<decltype(inner)>>) {
            return const_view_type_variant{inner};
          } else {
            return const_view_type_variant{inner.view()};
          }
        },
        data_);
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
  [[nodiscard]] auto view() { return view<memory_type_constant<>>(); }
  [[nodiscard]] auto view() const { return view<memory_type_constant<>>(); }
};

}  // namespace raft
