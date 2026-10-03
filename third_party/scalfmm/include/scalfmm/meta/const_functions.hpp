// --------------------------------
// See LICENCE file at project root
// File : scalfmm/meta/const_functions.hpp
// --------------------------------
#ifndef SCALFMM_META_CONST_FUNCTIONS_HPP
#define SCALFMM_META_CONST_FUNCTIONS_HPP

#include <cstddef>
#include <limits>
#include <type_traits>

namespace scalfmm::meta
{
    /**
     * @brief
     *
     * @tparam T
     * @tparam U
     * @param a
     * @param b
     * @param epsilon
     * @return true
     * @return false
     */
    template<typename T, typename U, typename = std::enable_if_t<std::is_floating_point<T>::value, T>,
             typename = std::enable_if_t<std::is_floating_point<U>::value, U>>
    constexpr auto feq(const T& a, const U& b, T epsilon = std::numeric_limits<T>::epsilon()) -> bool
    {
        return a - b < epsilon && b - a < epsilon;
    }

    /**
     * @brief
     *
     * @tparam T
     * @param a
     * @param p
     * @return T
     */
    template<typename T>
    constexpr auto pow(T a, std::size_t p) -> T
    {
        return p == 0 ? 1 : a * pow<T>(a, p - 1);
    }
}   // namespace scalfmm::meta

#endif
