// --------------------------------
// See LICENCE file at project root
// File : scalfmm/meta/type_pack.hpp
// --------------------------------
#ifndef SCALFMM_META_TYPE_PACK_HPP
#define SCALFMM_META_TYPE_PACK_HPP

#include <cstddef>
#include <tuple>

namespace scalfmm::meta
{
    /**
     * @brief
     *
     * @tparam I
     * @tparam T
     */
    template<std::size_t I, typename T>
    struct pack
    {
        enum
        {
            size = I
        };
        using type = T;
    };

    namespace details
    {
        /**
         * @brief
         *
         * @tparam Final
         * @tparam ExpandedTuple
         * @tparam ToExpand
         */
        template<template<class...> class Final, typename ExpandedTuple, typename... ToExpand>
        struct pack_expand_impl;

        /**
         * @brief
         *
         * @tparam Final
         * @tparam Expanded
         */
        template<template<class...> class Final, typename... Expanded>
        struct pack_expand_impl<Final, std::tuple<Expanded...>>
        {
            using type = Final<Expanded...>;
        };

        /**
         * @brief
         *
         * @tparam Final
         * @tparam T
         * @tparam Args
         * @tparam Expanded
         */
        template<template<class...> class Final, typename T, typename... Args, typename... Expanded>
        struct pack_expand_impl<Final, std::tuple<Expanded...>, pack<0, T>, Args...>
        {
            using type = typename pack_expand_impl<Final, std::tuple<Expanded...>, Args...>::type;
        };

        /**
         * @brief
         *
         * @tparam Final
         * @tparam count
         * @tparam T
         * @tparam Args
         * @tparam Expanded
         */
        template<template<class...> class Final, std::size_t count, typename T, typename... Args, typename... Expanded>
        struct pack_expand_impl<Final, std::tuple<Expanded...>, pack<count, T>, Args...>
        {
            using type =
              typename pack_expand_impl<Final, std::tuple<Expanded..., T>, pack<count - 1, T>, Args...>::type;
        };

        /**
         * @brief
         *
         * @tparam Final
         * @tparam T
         * @tparam Args
         * @tparam Expanded
         */
        template<template<class...> class Final, typename T, typename... Args, typename... Expanded>
        struct pack_expand_impl<Final, std::tuple<Expanded...>, T, Args...>
        {
            using type = typename pack_expand_impl<Final, std::tuple<Expanded..., T>, Args...>::type;
        };
    }   // namespace details

    /**
     * @brief
     *
     * @tparam VariadicType
     * @tparam TypePacks
     */
    template<template<class...> class VariadicType, typename... TypePacks>
    using pack_expand = typename details::pack_expand_impl<VariadicType, std::tuple<>, TypePacks...>::type;

    /**
     * @brief
     *
     * @tparam TypePacks
     */
    template<typename... TypePacks>
    using pack_expand_tuple = pack_expand<std::tuple, TypePacks...>;

}   // end namespace scalfmm::meta

#endif   // SCALFMM_META_TYPE_PACK_HPP
