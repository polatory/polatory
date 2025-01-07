#pragma once

namespace polatory::fmm {

struct InterpolatorConfiguration {
  static constexpr int kClassic = -1;

  // The order of the uniform interpolator.
  int order{};

  // The type of the uniform interpolator. d must be one of:
  //   kClassic: the polynomial interpolant,
  //   0, 1, ..., order - 1: the Floater-Hormann's rational interpolant of degree d.
  int d{};
};

inline bool operator==(const InterpolatorConfiguration& lhs, const InterpolatorConfiguration& rhs) {
  return lhs.order == rhs.order && lhs.d == rhs.d;
}

inline bool operator!=(const InterpolatorConfiguration& lhs, const InterpolatorConfiguration& rhs) {
  return !(lhs == rhs);
}

}  // namespace polatory::fmm
