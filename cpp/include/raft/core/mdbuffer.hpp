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

#include <algorithm>
#include <execution>
#include <optional>
// TODO(wphicks): Correctly handle cuh/hpp split
#include <raft/core/copy.cuh>
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
#include <raft/util/cudart_utils.hpp>
#include <thrust/device_ptr.h>
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

template <typename T, typename ContainerPolicyVariant = default_container_policy_variant<T>>
struct universal_buffer_reference {
  using value_type    = typename std::remove_cv_t<T>;
  using pointer       = value_type*;
  using const_pointer = value_type const*;

  universal_buffer_reference(pointer ptr,
                             memory_type mem_type,
                             stream_view stream = stream_view_per_thread)
    : ptr_{ptr}, mem_type_{mem_type}, stream_{stream}
  {
  }

#ifndef RAFT_DISABLE_CUDA
  explicit universal_buffer_reference(thrust::device_ptr<T> ptr,
                                      memory_type mem_type = memory_type::device,
                                      stream_view stream   = stream_view_per_thread)
    : universal_buffer_reference{ptr.get(), mem_type, stream}
  {
    RAFT_EXPECTS(is_device_accessible(mem_type),
                 "Attempted to create host-only reference from Thrust device pointer");
  }
#endif

  operator value_type() const  // NOLINT
  {
    auto result = value_type{};
    if (is_host_accessible(mem_type_)) {
      result = *ptr_;
    } else {
#ifdef RAFT_DISABLE_CUDA
      throw non_cuda_build_error{"Attempted to access device reference in non-CUDA build"};
#else
      update_host(&result, ptr_, 1, stream_);
#endif
    }
    return result;
  }

  auto operator=(value_type const& other) -> universal_buffer_reference<T, ContainerPolicyVariant>&
  {
    if (is_host_accessible(mem_type_)) {
      *ptr_ = other;
    } else {
#ifdef RAFT_DISABLE_CUDA
      throw non_cuda_build_error{"Attempted to assign to device reference in non-CUDA build"};
#else
      update_device(ptr_, &other, 1, stream_);
#endif
    }
    return *this;
  }

 private:
  pointer ptr_;
  raft::memory_type mem_type_;
  raft::stream_view stream_;
};

template <typename ElementType,
          typename ContainerPolicyVariant = default_container_policy_variant<ElementType>>
struct default_buffer_container_policy {
  using element_type = ElementType;
  using value_type   = std::remove_cv_t<element_type>;

  using reference       = universal_buffer_reference<element_type, ContainerPolicyVariant>;
  using const_reference = universal_buffer_reference<element_type const, ContainerPolicyVariant>;
  using pointer         = element_type*;
  using const_pointer   = element_type const*;

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
  template <raft::memory_type MemType,
            std::enable_if_t<has_stream<container_type<MemType>>()>* = nullptr>
  [[nodiscard]] auto constexpr access(container_type<MemType>& c, std::size_t n) const noexcept
  {
    return reference{c.data() + n, MemType, c.stream()};
  }

  template <raft::memory_type MemType,
            std::enable_if_t<!has_stream<container_type<MemType>>()>* = nullptr>
  [[nodiscard]] auto constexpr access(container_type<MemType>& c, std::size_t n) const noexcept
  {
    return reference{c.data() + n, MemType};
  }

  template <raft::memory_type MemType,
            std::enable_if_t<has_stream<container_type<MemType>>()>* = nullptr>
  [[nodiscard]] auto constexpr access(container_type<MemType> const& c,
                                      std::size_t n) const noexcept
  {
    return const_reference{c.data() + n, MemType, c.stream()};
  }

  template <raft::memory_type MemType,
            std::enable_if_t<!has_stream<container_type<MemType>>()>* = nullptr>
  [[nodiscard]] auto constexpr access(container_type<MemType> const& c,
                                      std::size_t n) const noexcept
  {
    return const_reference{c.data() + n, MemType};
  }

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

  using pointer         = typename container_policy_type::pointer;
  using const_pointer   = typename container_policy_type::const_pointer;
  using reference       = typename container_policy_type::reference;
  using const_reference = typename container_policy_type::const_reference;

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

    using from_mdspan_type_variant =
      std::conditional_t<from_has_mdspan_view,
                         std::conditional_t<is_variant_of_mdspans_v<std::declval<FromT>().view()>,
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
      is_copyable_memory_combination<memory_type::host, from_mdspan_type<SrcMemType>> ||
      is_copyable_memory_combination<memory_type::device, from_mdspan_type<SrcMemType>> ||
      is_copyable_memory_combination<memory_type::managed, from_mdspan_type<SrcMemType>> ||
      is_copyable_memory_combination<memory_type::pinned, from_mdspan_type<SrcMemType>>;

    // Note: This is the most generic possible test for constructibility, but
    // in practice, we may be satisfied with a check solely for
    // constructibility from matching memory types.
    auto static constexpr const value = is_copyable_to_any_memory_type<memory_type::host> ||
                                        is_copyable_to_any_memory_type<memory_type::device> ||
                                        is_copyable_to_any_memory_type<memory_type::managed> ||
                                        is_copyable_to_any_memory_type<memory_type::pinned>;

    using type = std::enable_if_t<value, T>;

    template <
      typename U = FromT,
      std::enable_if_t<std::conjunction_v<std::is_same_v<U, FromT>, from_has_mdspan_view, value>>* =
        nullptr>
    auto static constexpr get_mdspan(U&& from) -> from_mdspan_type_variant
    {
      return from.view();
    }

    template <typename U = FromT,
              std::enable_if_t<
                std::conjunction_v<std::is_same_v<U, FromT>, is_mdspan_v<U>, value>>* = nullptr>
    auto static constexpr const get_mdspan(U&& from)
    {
      return std::forward<U>(from);
    }
  };

  template <typename FromT, typename T = void>
  using constructible_from_t = typename constructible_from<true, FromT, T>::type

    constexpr mdbuffer() = default;

  template <typename FromT, constructible_from_t<FromT>* = nullptr>
  constexpr mdbuffer(raft::resources const& res,
                     FromT&& other,
                     memory_type mem_type =
                       constructible_from<true, FromT, memory_type>::default_mem_type_destination)
    : data_{[res, &other, mem_type]() {
        using config = constructible_from<true, FromT, void>;
        if constexpr (std::is_convertible_v<std::forward<FromT>, storage_type_variant>) {
          return storage_type_variant{std::move(other)};
        } else {
          auto result = storage_type_variant{[res, mem_type]() {
            switch (mem_type) {
              case memory_type::host: return owning_type<memory_type::host>{};
              case memory_type::device: return owning_type<memory_type::device>{};
            }
          }()};
          // TODO(wphicks): Construct owning variant of correct memory type. Copy
          // from other's view (remembering that it may be a variant of views).
          // Logic should be same for copy constructor.
        }
      }()}
  {
  }

  [[nodiscard]] auto constexpr mem_type()
  {
    return static_cast<memory_type>(data_.index() % std::variant_size_v<owning_type_variant>);
  };
  [[nodiscard]] auto constexpr is_owning()
  {
    return data_.index() >= std::variant_size_v<view_type_variant>;
  };
  [[nodiscard]] auto constexpr data_handle()
  {
    return fast_visit(
      [](auto&& inner) {
        if constexpr (std::is_convertible_v<decltype(inner.data_handle()), pointer>) {
          return pointer{inner.data_handle()};
        } else {
          return pointer{inner.data_handle().get()};
        }
      },
      data_);
  };
  [[nodiscard]] auto constexpr data_handle() const
  {
    return fast_visit(
      [](auto&& inner) {
        if constexpr (std::is_convertible_v<decltype(inner.data_handle()), const_pointer>) {
          return const_pointer{inner.data_handle()};
        } else {
          return const_pointer{inner.data_handle().get()};
        }
      },
      data_);
  }

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

 public:
  // TODO(wphicks): Allow this to take an optional memory type template
  // parameter and return non-variant view if available for that memory type.
  [[nodiscard]] auto view()
  {
    return fast_visit([](auto&& inner) { return get_view_from_data(inner); }, data_);
  }

 private:
  storage_type_variant data_{};
};

}  // namespace raft
