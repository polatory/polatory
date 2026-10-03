// --------------------------------
// See LICENCE file at project root
// File : scalfmm/simd/memory.hpp
// --------------------------------
#ifndef SCALFMM_SIMD_MEMORY_HPP
#define SCALFMM_SIMD_MEMORY_HPP

#include "scalfmm/meta/traits.hpp"
#include "scalfmm/meta/utils.hpp"

#include "xsimd/xsimd.hpp"

#include <cstddef>
#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace scalfmm::simd
{
    /**
     * @brief
     *
     */
    struct aligned
    {
    };

    /**
     * @brief
     *
     */
    struct unaligned
    {
    };

    /**
     * @brief
     *
     */
    struct splated
    {
    };

    namespace impl
    {
        /**
         * @brief
         *
         * @tparam SimdType
         * @tparam Tuple
         * @tparam Is
         * @param values_to_splat
         * @param s
         * @return constexpr auto
         */
        template<typename SimdType, typename Tuple, std::size_t... Is>
        [[nodiscard]] constexpr inline auto load_splat_value(Tuple const& values_to_splat, std::index_sequence<Is...> s)
        {
            return std::make_tuple(SimdType(meta::get<Is>(values_to_splat))...);
        }

        /**
         * @brief
         *
         * @tparam SimdType
         * @tparam TupleOfIt
         * @tparam Is
         * @param values_to_splat
         * @param s
         * @return constexpr auto
         */
        template<typename SimdType, typename TupleOfIt, std::size_t... Is>
        [[nodiscard]] constexpr inline auto load_splat_value_from_it(TupleOfIt const& values_to_splat,
                                                                     std::index_sequence<Is...> s)
        {
            return std::make_tuple(SimdType(*meta::get<Is>(values_to_splat))...);
        }

        /**
         * @brief
         *
         * @tparam Position
         * @tparam TuplePtr
         * @tparam Aligned
         * @tparam Is
         * @param particle_ptrs
         * @param a
         * @param s
         * @return constexpr auto
         */
        template<typename Position, typename TuplePtr, typename Aligned, std::size_t... Is>
        [[nodiscard]] constexpr inline auto load_position(TuplePtr const& particle_ptrs, Aligned a,
                                                          std::index_sequence<Is...> s)
        {
            using value_type = typename Position::value_type;
            if constexpr(meta::is_simd<value_type>::value)
            {
                if constexpr(std::is_same_v<Aligned, aligned>)
                {
                    return Position{xsimd::load_aligned(&meta::get<Is>(*particle_ptrs))...};
                }
                else if constexpr(std::is_same_v<Aligned, unaligned>)
                {
                    return Position{xsimd::load_unaligned(&meta::get<Is>(*particle_ptrs))...};
                }
                else if constexpr(std::is_same_v<Aligned, splated>)
                {
                    return Position{value_type(meta::get<Is>(*particle_ptrs))...};
                }
            }
            else
            {
                return Position{meta::get<Is>(*particle_ptrs)...};
            }
        }

        /**
         * @brief
         *
         * @tparam StoredType
         * @tparam Tuple
         * @tparam TuplePtr
         * @tparam Aligned
         * @tparam Is
         * @param variadic_adaptor_iterator
         * @param src
         * @param a
         * @param s
         */
        template<typename StoredType, typename Tuple, typename TuplePtr, typename Aligned, std::size_t... Is>
        constexpr inline void store_tuple(TuplePtr& variadic_adaptor_iterator, Tuple const& src, Aligned a,
                                          std::index_sequence<Is...> s)
        {
            if constexpr(meta::is_simd<StoredType>::value)
            {
                if constexpr(std::is_same_v<Aligned, aligned>)
                {
                    meta::noop_f{
                      (xsimd::store_aligned(&meta::get<Is>(*variadic_adaptor_iterator), meta::get<Is>(src)), 0)...};
                }
                else if constexpr(std::is_same_v<Aligned, unaligned>)
                {
                    meta::noop_f{
                      (xsimd::store_unaligned(&meta::get<Is>(*variadic_adaptor_iterator), meta::get<Is>(src)), 0)...};
                }
            }
            else
            {
                meta::noop_f{(meta::get<Is>(*variadic_adaptor_iterator) = meta::get<Is>(src), 0)...};
            }
        }

        /**
         * @brief
         *
         * @tparam LoadedType
         * @tparam TuplePtr
         * @tparam Aligned
         * @tparam Is
         * @param particle_ptrs
         * @param a
         * @param s
         * @return constexpr auto
         */
        template<typename LoadedType, typename TuplePtr, typename Aligned, std::size_t... Is>
        [[nodiscard]] constexpr inline auto load_tuple(TuplePtr const& particle_ptrs, Aligned a,
                                                       std::index_sequence<Is...> s)
        {
            if constexpr(meta::is_simd<LoadedType>::value)
            {
                if constexpr(std::is_same_v<Aligned, aligned>)
                {
                    return std::make_tuple(xsimd::load_aligned(&meta::get<Is>(*particle_ptrs))...);
                }
                else if constexpr(std::is_same_v<Aligned, unaligned>)
                {
                    return std::make_tuple(xsimd::load_unaligned(&meta::get<Is>(*particle_ptrs))...);
                }
                else if constexpr(std::is_same_v<Aligned, splated>)
                {
                    return std::make_tuple(LoadedType(meta::get<Is>(*particle_ptrs))...);
                }
            }
            else
            {
                return std::make_tuple(meta::get<Is>(*particle_ptrs)...);
            }
        }

        /**
         * @brief
         *
         * @tparam F
         * @tparam Container
         * @tparam Types
         * @tparam Is
         * @param s
         * @param f
         * @param input
         * @param ts
         * @return constexpr auto
         */
        template<typename F, typename Container, typename... Types, std::size_t... Is>
        [[nodiscard]] constexpr inline auto apply_f(std::index_sequence<Is...> s, F&& f, Container&& input,
                                                    Types&&... ts)
        {
            using return_type = typename std::decay_t<Container>;
            return return_type{{std::invoke(std::forward<F>(f), meta::get<Is>(std::forward<Container>(input)),
                                            std::forward<Types>(ts)...)...}};
        }

    }   // namespace impl

    /**
     * @brief
     *
     * @tparam SimdType
     * @tparam Tuple
     * @param values_to_splat
     * @return constexpr auto
     */
    template<typename SimdType, typename Tuple>
    [[nodiscard]] constexpr inline auto load_splat_value(Tuple const& values_to_splat)
    {
        return impl::load_splat_value<SimdType>(values_to_splat, std::make_index_sequence<meta::tuple_size_v<Tuple>>{});
    }

    /**
     * @brief
     *
     * @tparam SimdType
     * @tparam TupleOfIt
     * @param values_to_splat
     * @return constexpr auto
     */
    template<typename SimdType, typename TupleOfIt>
    [[nodiscard]] constexpr inline auto load_splat_value_from_it(TupleOfIt const& values_to_splat)
    {
        return impl::load_splat_value_from_it<SimdType>(values_to_splat,
                                                        std::make_index_sequence<meta::tuple_size_v<TupleOfIt>>{});
    }

    /**
     * @brief
     *
     * @tparam Position
     * @tparam TuplePtr
     * @tparam Aligned
     * @param particle_ptrs
     * @param a
     * @return constexpr auto
     */
    template<typename Position, typename TuplePtr, typename Aligned = aligned>
    [[nodiscard]] constexpr inline auto load_position(TuplePtr const& particle_ptrs, Aligned a = Aligned{})
    {
        return impl::load_position<Position>(particle_ptrs, a, std::make_index_sequence<Position::dimension>{});
    }

    /**
     * @brief
     *
     * @tparam LoadedType
     * @tparam TuplePtr
     * @tparam Aligned
     * @param variadic_adaptor_iterator
     * @param a
     * @return constexpr auto
     */
    template<typename LoadedType, typename TuplePtr, typename Aligned = aligned>
    [[nodiscard]] constexpr inline auto load_tuple(TuplePtr const& variadic_adaptor_iterator, Aligned a = Aligned{})
    {
        return impl::load_tuple<LoadedType>(
          variadic_adaptor_iterator, a,
          std::make_index_sequence<meta::tuple_size_v<decltype(std::declval<TuplePtr>().operator*())>>{});
    }

    /**
     * @brief
     *
     * @tparam Position
     * @tparam TuplePtr
     * @tparam Aligned
     * @param particle_ptrs
     * @param src
     * @param a
     */
    template<typename Position, typename TuplePtr, typename Aligned = aligned>
    constexpr inline void store_position(TuplePtr& particle_ptrs, Position const& src, Aligned a = Aligned{})
    {
        impl::store_tuple<typename Position::value_type>(particle_ptrs, src, a,
                                                         std::make_index_sequence<Position::dimension>{});
    }

    /**
     * @brief
     *
     * @tparam StoredType
     * @tparam TupleSrc
     * @tparam TuplePtr
     * @tparam Aligned
     * @param variadic_adaptor_iterator
     * @param src
     * @param a
     */
    template<typename StoredType, typename TupleSrc, typename TuplePtr, typename Aligned = aligned>
    constexpr inline void store_tuple(TuplePtr& variadic_adaptor_iterator, TupleSrc const& src, Aligned a = Aligned{})
    {
        impl::store_tuple<StoredType>(variadic_adaptor_iterator, src, a,
                                      std::make_index_sequence<meta::tuple_size_v<TupleSrc>>{});
    }

    /**
     * @brief
     *
     * @tparam Size
     * @tparam F
     * @tparam Container
     * @tparam Types
     * @param f
     * @param input
     * @param ts
     * @return constexpr auto
     */
    template<std::size_t Size, typename F, typename Container, typename... Types>
    [[nodiscard]] constexpr inline auto apply_f(F&& f, Container&& input, Types&&... ts)
    {
        return impl::apply_f(std::make_index_sequence<Size>{}, std::forward<F>(f), std::forward<Container>(input),
                             std::forward<Types>(ts)...);
    }

}   // namespace scalfmm::simd

#endif   // SCALFMM_SIMD_MEMORY_HPP
