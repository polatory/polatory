#pragma once

#include <boost/container_hash/hash.hpp>
#include <functional>

namespace polatory::fmm {

struct InterpolatorConfiguration {
  static constexpr int kClassic = -1;

  int tree_height{};

  // The order of the uniform interpolator.
  int order{};

  // The type of the uniform interpolator. d must be one of:
  //   kClassic: the polynomial interpolant,
  //   0, 1, ..., order - 1: the Floater-Hormann's rational interpolant of degree d.
  int d{};
};

inline bool operator==(const InterpolatorConfiguration& lhs, const InterpolatorConfiguration& rhs) {
  return lhs.tree_height == rhs.tree_height && lhs.order == rhs.order && lhs.d == rhs.d;
}

inline bool operator!=(const InterpolatorConfiguration& lhs, const InterpolatorConfiguration& rhs) {
  return !(lhs == rhs);
}

}  // namespace polatory::fmm

template <>
struct std::hash<polatory::fmm::InterpolatorConfiguration> {
  std::size_t operator()(const polatory::fmm::InterpolatorConfiguration& config) const noexcept {
    std::size_t seed{};
    boost::hash_combine(seed, config.tree_height);
    boost::hash_combine(seed, config.order);
    boost::hash_combine(seed, config.d);
    return seed;
  }
};
