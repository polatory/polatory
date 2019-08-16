// Copyright (c) 2016, GSI and The Polatory Authors.

#pragma once

#include <utility>

namespace polatory {
namespace common {

template <class BinaryFunction, class T>
auto fold_left(BinaryFunction /*f*/, const T& x) {
  return x;
}

template <class BinaryFunction, class T1, class T2, class... Ts>
auto fold_left(BinaryFunction f, T1&& x1, T2&& x2, Ts&&... xs) {
  return fold_left(f, f(std::forward<T1>(x1), std::forward<T2>(x2)), std::forward<Ts>(xs)...);
}

template <class BinaryFunction, class T>
auto fold_right(BinaryFunction /*f*/, const T& x) {
  return x;
}

template <class BinaryFunction, class T1, class T2, class... Ts>
auto fold_right(BinaryFunction f, T1&& x1, T2&& x2, Ts&&... xs) {
  return f(std::forward<T1>(x1), fold_right(f, std::forward<T2>(x2), std::forward<Ts>(xs)...));
}

}  // namespace common
}  // namespace polatory
