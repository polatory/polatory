#pragma once

namespace polatory::fmm {

struct interpolator_configuration {
  static constexpr int kClassic = -1;

  int order{};

  // kClassic: use the classic uniform interpolator
  // 0, 1, ..., order - 1: use the modified uniform interpolator
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
