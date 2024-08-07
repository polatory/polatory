#pragma once

#include <array>

namespace polatory::isosurface::rmt {

// Edge index per node: 0 - 13
using edge_index = int;

struct edge {
  static constexpr edge_index k0 = 0;
  static constexpr edge_index k1 = 1;
  static constexpr edge_index k2 = 2;
  static constexpr edge_index k3 = 3;
  static constexpr edge_index k4 = 4;
  static constexpr edge_index k5 = 5;
  static constexpr edge_index k6 = 6;
  static constexpr edge_index k7 = 7;
  static constexpr edge_index k8 = 8;
  static constexpr edge_index k9 = 9;
  static constexpr edge_index kA = 10;
  static constexpr edge_index kB = 11;
  static constexpr edge_index kC = 12;
  static constexpr edge_index kD = 13;
};

inline const std::array<edge_index, 14> kOppositeEdge{
    edge::k7, edge::k8, edge::k9, edge::kA, edge::kB, edge::kC, edge::kD,
    edge::k0, edge::k1, edge::k2, edge::k3, edge::k4, edge::k5, edge::k6};

}  // namespace polatory::isosurface::rmt
