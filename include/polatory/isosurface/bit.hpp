// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <type_traits>

namespace polatory {
namespace isosurface {

namespace detail {

inline
int naive_ctz(int x) {
  int count = 0;
  while (!(x & 1)) {
    x >>= 1;
    count++;
  }
  return count;
}

inline
int naive_popcnt(int x) {
  int count = 0;
  while (x) {
    x &= x - 1;
    count++;
  }
  return count;
}

} // namespace detail

template <class Integral, typename std::enable_if<
  std::is_integral<Integral>::value && sizeof(Integral) <= sizeof(int)
  , std::nullptr_t>::type = nullptr>
int bit_count(Integral bit_set) {
#if defined(__INTEL_COMPILER)
  return _popcnt32(bit_set);
#elif defined(__GNUC__)
  return __builtin_popcount(bit_set);
#else
  return detail::naive_popcnt(bit_set);
#endif
}

template <class Integral, typename std::enable_if<
  std::is_integral<Integral>::value && sizeof(Integral) <= sizeof(int)
  , std::nullptr_t>::type = nullptr>
int bit_peek(Integral bit_set) {
  if (bit_set == 0) return -1;

#if defined(__INTEL_COMPILER)
  return _bit_scan_forward(bit_set);
#elif defined(__GNUC__)
  return __builtin_ctz(bit_set);
#else
  return detail::naive_ctz(bit_set);
#endif
}

template <class Integral, typename std::enable_if<
  std::is_integral<Integral>::value && sizeof(Integral) <= sizeof(int)
  , std::nullptr_t>::type = nullptr>
int bit_pop(Integral *bit_set) {
  if (*bit_set == 0) return -1;

  int bit_idx = bit_peek(*bit_set);
  int bit = 1 << bit_idx;
  *bit_set ^= bit;

  return bit_idx;
}

} // namespace isosurface
} // namespace polatory
