// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#ifdef _MSC_VER
#include <intrin.h>
#endif
#include <type_traits>

namespace polatory {
namespace isosurface {

namespace detail {

inline
int naive_ctz(int x) {
  int count = 0;
  while ((x & 1) == 0) {
    x >>= 1;
    count++;
  }
  return count;
}

inline
int naive_popcnt(int x) {
  int count = 0;
  while (x != 0) {
    x &= x - 1;
    count++;
  }
  return count;
}

}  // namespace detail

template <class Integral, typename std::enable_if<
  std::is_integral<Integral>::value && sizeof(Integral) <= sizeof(int)
  , std::nullptr_t>::type = nullptr>
int bit_count(Integral bit_set) {
#if defined(__INTEL_COMPILER)
  return _popcnt32(bit_set);
#elif defined(__GNUC__)
  return __builtin_popcount(static_cast<unsigned int>(bit_set));
#elif defined(_MSC_VER)
  return static_cast<int>(__popcnt(static_cast<unsigned int>(bit_set)));
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
  return __builtin_ctz(static_cast<unsigned int>(bit_set));
#elif defined(_MSC_VER)
  unsigned long bit_index;  // NOLINT(runtime/int)
  _BitScanForward(&bit_index, static_cast<unsigned long>(bit_set));  // NOLINT(runtime/int)
  return static_cast<int>(bit_index);
#else
  return detail::naive_ctz(bit_set);
#endif
}

template <class Integral, typename std::enable_if<
  std::is_integral<Integral>::value && sizeof(Integral) <= sizeof(int)
  , std::nullptr_t>::type = nullptr>
int bit_pop(Integral *bit_set) {
  if (*bit_set == 0) return -1;

  int bit_index = bit_peek(*bit_set);
  int bit = 1 << bit_index;
  *bit_set ^= bit;

  return bit_index;
}

}  // namespace isosurface
}  // namespace polatory
