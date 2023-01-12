#pragma once

#include <cmath>

namespace polatory {
namespace common {

template <class T>
T pi() {
  return T(4.0) * std::atan(T(1.0));
}

}  // namespace common
}  // namespace polatory
