// --------------------------------
// See LICENCE file at project root
// File : scalfmm/container/simple_variadic.hpp
// --------------------------------
#ifndef SCALFMM_CONTAINER_SIMPLE_VARIADIC_CONTAINER_HPP
#define SCALFMM_CONTAINER_SIMPLE_VARIADIC_CONTAINER_HPP

#include "xsimd/config/xsimd_config.hpp"

#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace scalfmm::container
{
    /**
     * @brief Forward declaration
     *
     * @tparam Types
     * @tparam Indices
     */
    template<typename Types, typename Indices>
    class variad_impl;

    /**
     * @brief
     *
     * @tparam Ts
     * @tparam Indices
     */
    template<typename... Ts, std::size_t... Indices>
    class variad_impl<std::tuple<Ts...>, std::integer_sequence<std::size_t, Indices...>>
      : public std::tuple<std::vector<Ts, XSIMD_DEFAULT_ALLOCATOR(Ts)>...>
    {
      private:
        /**
         * @brief Discard fold expression results.
         *
         */
        struct noop_t
        {
            template<typename... Types>
            noop_t(const Types&... ts)
            {
            }
        };

      public:
        using base_type = std::tuple<std::vector<Ts, XSIMD_DEFAULT_ALLOCATOR(Ts)>...>;
        using value_type = std::tuple<Ts...>;
        using allocator_type = std::tuple<XSIMD_DEFAULT_ALLOCATOR(Ts)...>;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using reference = std::tuple<Ts&...>&;
        using const_reference = const std::tuple<const Ts&...>&;
        using pointer = std::tuple<Ts*...>;
        using const_pointer = const std::tuple<const Ts*...>;

      private:
        /**
         * @brief
         *
         */
        allocator_type m_allocator{};

      public:
        /**
         * @brief Construct a new variad impl object
         *
         */
        variad_impl() = default;

        /**
         * @brief Construct a new variad impl object
         *
         */
        variad_impl(const variad_impl&) = default;

        /**
         * @brief Construct a new variad impl object
         *
         */
        variad_impl(variad_impl&&) noexcept = default;

        /**
         * @brief
         *
         * @return variad_impl&
         */
        variad_impl& operator=(const variad_impl&) = default;

        /**
         * @brief
         *
         * @return variad_impl&
         */
        variad_impl& operator=(variad_impl&&) noexcept = default;

        /**
         * @brief Destroy the variad impl object
         *
         */
        ~variad_impl() = default;

        // TODO
        // explicit variad( const allocator_type& alloc){}

        /**
         * @brief Construct a new variad impl object
         *
         * @param count
         * @param value
         * @param alloc
         */
        explicit variad_impl(size_type count, const value_type& value, const allocator_type& alloc = allocator_type())
          : m_allocator(alloc)
        {
            reserve(count);
            for(std::size_t i = 0; i < count; ++i)
            {
                push_back(value);
            }
        }

        /**
         * @brief Construct a new variad impl object
         *
         * @param count
         */
        explicit variad_impl(size_type count)
        {
            reserve(count);
            for(std::size_t i = 0; i < count; ++i)
            {
                push_back(value_type{});
            }
        }

        /**
         * @brief
         *
         * @param size
         */
        inline void reserve(size_type size) { noop_t{(std::get<Indices>(*this).reserve(size), 0)...}; }

        /**
         * @brief
         *
         * @param value
         */
        inline void push_back(const value_type& value)
        {
            noop_t{(std::get<Indices>(*this).push_back(std::get<Indices>(value)), 0)...};
        }
    };

    /**
     * @brief
     *
     * @tparam Ts
     */
    template<typename... Ts>
    struct variad : public variad_impl<std::tuple<Ts...>, std::make_index_sequence<sizeof...(Ts)>>
    {
        using base_type = variad_impl<std::tuple<Ts...>, std::make_index_sequence<sizeof...(Ts)>>;
        using base_type::base_type;
    };
}   // namespace scalfmm::container
#endif   // SCALFMM_CONTAINER_SIMPLE_VARIADIC_CONTAINER_HPP
