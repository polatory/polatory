#pragma once

#include <bit>

namespace polatory::isosurface {

template <class T>
int bit_count(T bit_set) {
  return std::popcount(bit_set);
}

template <class T>
int bit_peek(T bit_set) {
  if (bit_set == 0) {
    return -1;
  }

  return std::countr_zero(bit_set);
}

template <class T>
int bit_pop(T *bit_set) {
  if (*bit_set == 0) {
    return -1;
  }

  auto bit_index = bit_peek(*bit_set);
  auto bit = T{1} << bit_index;
  *bit_set ^= bit;

  return bit_index;
}

}  // namespace polatory::isosurface
