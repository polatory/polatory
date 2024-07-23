#pragma once

#include <polatory/common/io.hpp>

namespace polatory::fmm {

struct interpolator_configuration {
  static constexpr int kClassic = -1;

  // The order of the uniform interpolator.
  int order{};

  // The type of the uniform interpolator. d must be one of:
  //   kClassic: the polynomial interpolant,
  //   0, 1, ..., order - 1: the Floater-Hormann's rational interpolant of degree d.
  int d{};
};

inline bool operator==(const interpolator_configuration& lhs,
                       const interpolator_configuration& rhs) {
  return lhs.order == rhs.order && lhs.d == rhs.d;
}

inline bool operator!=(const interpolator_configuration& lhs,
                       const interpolator_configuration& rhs) {
  return !(lhs == rhs);
}

}  // namespace polatory::fmm

namespace polatory::common {

template <>
struct Read<fmm::interpolator_configuration> {
  void operator()(std::istream& is, fmm::interpolator_configuration& t) const {
    int order{};
    int d{};
    read(is, order);
    read(is, d);
    t = {order, d};
  }
};

template <>
struct Write<fmm::interpolator_configuration> {
  void operator()(std::ostream& os, const fmm::interpolator_configuration& t) const {
    write(os, t.order);
    write(os, t.d);
  }
};

}  // namespace polatory::common
