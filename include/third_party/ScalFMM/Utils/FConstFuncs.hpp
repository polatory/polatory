#ifndef SCALFMM_CONSTFUNCS
#define SCALFMM_CONSTFUNCS

#include <cstddef>
#include <type_traits>

template<typename T>
constexpr T Fpow(T a, std::size_t p) {
    return p == 0 ? 1 : a * Fpow<T>(a, p-1);
}

template<typename T, typename U,
         typename std::enable_if<std::is_arithmetic<T>::value, T>::type* = nullptr,
         typename std::enable_if<std::is_arithmetic<U>::value, U>::type* = nullptr>
constexpr bool Ffeq(T a, U b, T epsilon = 1e-7) {
    return a - b < epsilon && b - a < epsilon;
}


#endif
