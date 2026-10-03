// --------------------------------
// See LICENCE file at project root
// File : scalfmm/functional/utils.hpp
// --------------------------------
#ifndef SCALFMM_FUNCTIONAL_UTILS_HPP
#define SCALFMM_FUNCTIONAL_UTILS_HPP

#include "scalfmm/meta/traits.hpp"

#include <cmath>
#include <type_traits>

namespace scalfmm::utils
{
    /**
     * @brief Squared norm of a range.
     *
     * @tparam T
     * @param range
     * @return std::enable_if_t<meta::has_value_type<T>::value && meta::has_range_interface<T>::value, typename T::value_type>
     */
    template<typename T>
    inline auto norm2(const T& range)
      -> std::enable_if_t<meta::has_value_type<T>::value && meta::has_range_interface<T>::value, typename T::value_type>
    {
        typename T::value_type square_sum{0};
        for(auto a: range)
        {
            square_sum += a * a;
        }
        return square_sum;
    }

    /**
     * @brief Norm of a range.
     *
     * @tparam T
     * @param range
     * @return std::enable_if_t<meta::has_value_type<T>::value && meta::has_range_interface<T>::value, typename T::value_type>
     */
    template<typename T>
    inline auto norm(const T& range)
      -> std::enable_if_t<meta::has_value_type<T>::value && meta::has_range_interface<T>::value, typename T::value_type>
    {
        return std::sqrt(norm2(range));
    }

}   // namespace scalfmm::utils

#endif   // SCALFMM_FUNCTIONAL_UTILC_HPP
