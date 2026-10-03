// --------------------------------
// See LICENCE file at project root
// File : scalfmm/utils/math.hpp
// --------------------------------
#ifndef SCALFMM_UTILS_MATH_HPP
#define SCALFMM_UTILS_MATH_HPP

#include <cfloat>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace scalfmm::math
{
    /**
     * @brief compute the factoriel  of value
     *
     * @tparam T
     * 
     * @param value
     * 
     * @return T value!
     */
    template<typename T>
    inline auto factorial(int value) -> T
    {
        if(value == 0)
        {
            return T{1};
        }

        int result = value;
        while(--value > 1)
        {
            result *= value;
        }
        return T(result);
    }
    /**
     * @brief Compute a^p when p is an integer (meta function)
     *
     * @tparam T
     * @param a
     * @param p
     * @return T
     */
    template<typename T>
    inline constexpr auto pow(const T a, const std::size_t p) -> T
    {
        return p == 0 ? 1 : a * pow<T>(a, p - 1);
    }

    template<typename T>
    inline auto pow(T a, int p) -> T
    {
        T result{T(1.)};
        while(p-- > 0)
        {
            result *= a;
        }
        return result;
    }

    /**
     * @brief Check if |a b| < epsilon for two floating_point
     *
     * @tparam T
     * @tparam U
     * @param a
     * @param b
     * @param epsilon the threshold
     * @return 
     */
    template<typename T, typename U, typename = std::enable_if_t<std::is_floating_point<T>::value, T>,
             typename = std::enable_if_t<std::is_floating_point<U>::value, U>>
    inline constexpr auto feq(const T& a, const U& b, T epsilon = std::numeric_limits<T>::epsilon()) -> bool
    {
        return std::abs(a - b) < epsilon;
    }

    /**
     * @brief return true if value in [range_begin, range_end[
     *
     * @tparam ValueType1 for value
     * @tparam ValueType for the interval
     * @param value
     * @param range_begin
     * @param range_end
     * @return return true if value in [range_begin, range_end[
     *
     */
    template<typename ValueType1, typename ValueType>
    inline constexpr auto between(ValueType1 value, ValueType range_begin, ValueType range_end) -> bool
    {
        return (value >= range_begin && value < range_end);
    }

}   // namespace scalfmm::math

#endif   // SCALFMM_UTILS_MATH_HPP
