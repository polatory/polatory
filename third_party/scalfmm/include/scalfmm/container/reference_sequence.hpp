// --------------------------------
// See LICENCE file at project root
// File : scalfmm/container/reference_sequence.hpp
// --------------------------------
#ifndef SCALFMM_CONTAINER_REFERENCE_SEQUENCE_HPP
#define SCALFMM_CONTAINER_REFERENCE_SEQUENCE_HPP

#include "scalfmm/meta/traits.hpp"
#include "scalfmm/meta/utils.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

namespace scalfmm::container
{
    /**
     * @brief Overloads to get a generic function of returning reference.
     *
     * @tparam T
     * @param t
     * @return constexpr auto
     */
    template<typename T>
    constexpr inline auto generic_ref(T& t)
    {
        return std::ref(t);
    }

    /**
     * @brief Overloads to get a generic function of returning reference.
     *
     * @tparam T
     * @param t
     * @return constexpr auto
     */
    template<typename T>
    constexpr inline auto generic_ref(T const& t)
    {
        return std::cref(t);
    }

    /**
     * @brief Get the reference sequence impl object
     *
     * @tparam Seq
     * @tparam Is
     * @param s
     * @return constexpr auto
     */
    template<typename Seq, std::size_t... Is>
    constexpr inline auto get_reference_sequence_impl(Seq&& s, std::index_sequence<Is...>)
    {
        using seq_type = std::decay_t<Seq>;
        // if the sequence is an array
        if constexpr(meta::is_array_v<seq_type>)
        {
            // value_type depend if the seq reference is const or not
            // if it's const the reference wrapper store a const qualified type
            using value_type =
              std::conditional_t<std::is_const_v<std::remove_reference_t<Seq>>,
                                 std::add_const_t<typename seq_type::value_type>, typename seq_type::value_type>;
            using arr_ref_type = std::array<std::reference_wrapper<value_type>, sizeof...(Is)>;
            return arr_ref_type{generic_ref(meta::get<Is>(std::forward<Seq>(s)))...};
        }
        // if the sequence is a tuple not storing reference
        if constexpr(meta::is_tuple_v<seq_type> &&
                     meta::all(!std::is_lvalue_reference_v<std::tuple_element_t<Is, seq_type>>...))
        {
            using value_type = std::conditional_t<std::is_const_v<std::remove_reference_t<Seq>>,
                                                  std::add_const_t<std::tuple_element_t<0, seq_type>>,
                                                  std::tuple_element_t<0, seq_type>>;
            using arr_ref_type = std::array<std::reference_wrapper<value_type>, sizeof...(Is)>;
            return arr_ref_type{generic_ref(meta::get<Is>(std::forward<Seq>(s)))...};
        }
        // if the sequence is a tuple of reference
        if constexpr(meta::is_tuple_v<seq_type> &&
                     meta::all(std::is_lvalue_reference_v<std::tuple_element_t<Is, seq_type>>...))
        {
            using value_type = std::remove_reference_t<std::tuple_element_t<0, seq_type>>;
            using arr_ref_type = std::array<std::reference_wrapper<value_type>, sizeof...(Is)>;
            return arr_ref_type{meta::get<Is>(std::forward<Seq>(s))...};
        }
    }

    /**
     * @brief Get the reference sequence object
     *
     * @tparam Seq
     * @param s
     * @return constexpr auto
     */
    template<typename Seq>
    constexpr inline auto get_reference_sequence(Seq&& s)
    {
        return get_reference_sequence_impl(std::forward<Seq>(s),
                                           std::make_index_sequence<meta::tuple_size_v<std::decay_t<Seq>>>{});
    }
}   // namespace scalfmm::container

#endif   // SCALFMM_CONTAINER_REFERENCE_SEQUENCE_HPP
