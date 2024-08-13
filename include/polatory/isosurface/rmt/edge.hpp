#pragma once

#include <array>

namespace polatory::isosurface::rmt {

// Edge index per node: 0 - 13
using EdgeIndex = int;

struct Edge {
  static constexpr EdgeIndex k0 = 0;
  static constexpr EdgeIndex k1 = 1;
  static constexpr EdgeIndex k2 = 2;
  static constexpr EdgeIndex k3 = 3;
  static constexpr EdgeIndex k4 = 4;
  static constexpr EdgeIndex k5 = 5;
  static constexpr EdgeIndex k6 = 6;
  static constexpr EdgeIndex k7 = 7;
  static constexpr EdgeIndex k8 = 8;
  static constexpr EdgeIndex k9 = 9;
  static constexpr EdgeIndex kA = 10;
  static constexpr EdgeIndex kB = 11;
  static constexpr EdgeIndex kC = 12;
  static constexpr EdgeIndex kD = 13;
};

inline const std::array<EdgeIndex, 14> kOppositeEdge{
    Edge::k7, Edge::k8, Edge::k9, Edge::kA, Edge::kB, Edge::kC, Edge::kD,
    Edge::k0, Edge::k1, Edge::k2, Edge::k3, Edge::k4, Edge::k5, Edge::k6};

}  // namespace polatory::isosurface::rmt
