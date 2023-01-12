#pragma once

#include <functional>
#include <utility>

namespace polatory {
namespace common {

template <class T, class Compare = std::less<T>>
auto make_sorted_pair(T&& first, T&& second, Compare comp = Compare()) {
  return comp(first, second)
         ? std::make_pair(std::forward<T>(first), std::forward<T>(second))
         : std::make_pair(std::forward<T>(second), std::forward<T>(first));
}

}  // namespace common
}  // namespace polatory
