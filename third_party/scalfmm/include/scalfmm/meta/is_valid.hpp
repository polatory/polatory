//==============================================================================
// This file is a test for a clean scalfmm code.
// It's a attempt to a cleaner and more expressive code for the user.
// Author : Pierre Esterie
//==============================================================================
// --------------------------------
// See LICENCE file at project root
// File : scalfmm/meta/is_valid.hpp
// --------------------------------
#ifndef SCALFMM_META_IS_VALID_HPP
#define SCALFMM_META_IS_VALID_HPP

#include <type_traits>

namespace scalfmm::meta
{
    namespace details
    {
        /**
         * @brief
         *
         * @tparam F
         * @tparam Args
         * @tparam typename
         * @return constexpr auto
         */
        template<typename F, typename... Args, typename = decltype(std::declval<F&&>()(std::declval<Args&&>()...))>
        constexpr auto is_valid_impl(int)
        {
            return std::true_type{};
        }

        /**
         * @brief
         *
         * @tparam F
         * @tparam Args
         * @param ...
         * @return constexpr auto
         */
        template<typename F, typename... Args>
        constexpr auto is_valid_impl(...)
        {
            return std::false_type{};
        }

        /**
         * @brief
         *
         * @tparam F
         */
        template<typename F>
        struct is_valid_fun
        {
            /**
             * @brief
             *
             * @tparam Args
             * @return constexpr auto
             */
            template<typename... Args>
            constexpr auto operator()(Args&&...) const
            {
                return is_valid_impl<F, Args&&...>(int{});
            }
        };
    }   // namespace details

    /**
     * @brief
     *
     */
    struct is_valid_t
    {
        /**
         * @brief
         *
         * @tparam F
         * @return constexpr auto
         */
        template<typename F>
        constexpr auto operator()(F&&) const
        {
            return details::is_valid_fun<F&&>{};
        }

        /**
         * @brief
         *
         * @tparam F
         * @tparam Args
         * @param f
         * @param args
         * @return constexpr auto
         */
        template<typename F, typename... Args>
        constexpr auto operator()(F&& f, Args&&... args) const
        {
            return details::is_valid_impl<F&&, Args&&...>(int{});
        }
    };

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    struct type_w
    {
        using type = T;
    };

    /**
     * @brief
     *
     * @tparam T
     */
    template<typename T>
    constexpr type_w<T> type_c{};

    /**
     * @brief
     *
     */
    constexpr is_valid_t is_valid{};

    /**
     * @brief
     *
     * @tparam F
     * @tparam Args
     */
    template<typename F, typename... Args>
    inline constexpr bool is_valid_v = [](F&& f, Args&&... args) constexpr -> bool
    { return decltype(is_valid(f, args...))::value; }();

}   // namespace scalfmm::meta

#endif
