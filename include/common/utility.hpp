// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <utility>

namespace polatory {
namespace common {

template <class T, class Compare = std::less<T>>
auto make_sorted_pair(const T& first, const T& second, Compare comp = Compare()) {
  return comp(first, second)
         ? std::make_pair(first, second)
         : std::make_pair(second, first);
}

} // namespace common
} // namespace polatory
