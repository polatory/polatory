#pragma once

#include <array>
#include <polatory/isosurface/rmt/edge.hpp>
#include <polatory/isosurface/rmt/lattice_coordinates.hpp>

namespace polatory::isosurface::rmt {

inline const std::array<lattice_coordinates, 14> kNeighborLatticeCoordinatesDeltas{{
    lattice_coordinates(1, 0, 0),     // 0
    lattice_coordinates(1, 0, 1),     // 1
    lattice_coordinates(0, 0, 1),     // 2
    lattice_coordinates(1, 1, 0),     // 3
    lattice_coordinates(1, 1, 1),     // 4
    lattice_coordinates(0, 1, 1),     // 5
    lattice_coordinates(0, 1, 0),     // 6
    lattice_coordinates(-1, 0, 0),    // 7
    lattice_coordinates(-1, 0, -1),   // 8
    lattice_coordinates(0, 0, -1),    // 9
    lattice_coordinates(-1, -1, 0),   // A
    lattice_coordinates(-1, -1, -1),  // B
    lattice_coordinates(0, -1, -1),   // C
    lattice_coordinates(0, -1, 0),    // D
}};

inline lattice_coordinates neighbor(const lattice_coordinates& lc, edge_index ei) {
  return lc + kNeighborLatticeCoordinatesDeltas.at(ei);
}

}  // namespace polatory::isosurface::rmt
