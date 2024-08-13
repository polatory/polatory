#pragma once

#include <array>
#include <polatory/isosurface/rmt/edge.hpp>
#include <polatory/isosurface/rmt/lattice_coordinates.hpp>

namespace polatory::isosurface::rmt {

inline const std::array<LatticeCoordinates, 14> kNeighborLatticeCoordinatesDeltas{{
    LatticeCoordinates(1, 0, 0),     // 0
    LatticeCoordinates(1, 0, 1),     // 1
    LatticeCoordinates(0, 0, 1),     // 2
    LatticeCoordinates(1, 1, 0),     // 3
    LatticeCoordinates(1, 1, 1),     // 4
    LatticeCoordinates(0, 1, 1),     // 5
    LatticeCoordinates(0, 1, 0),     // 6
    LatticeCoordinates(-1, 0, 0),    // 7
    LatticeCoordinates(-1, 0, -1),   // 8
    LatticeCoordinates(0, 0, -1),    // 9
    LatticeCoordinates(-1, -1, 0),   // A
    LatticeCoordinates(-1, -1, -1),  // B
    LatticeCoordinates(0, -1, -1),   // C
    LatticeCoordinates(0, -1, 0),    // D
}};

inline LatticeCoordinates neighbor(const LatticeCoordinates& lc, EdgeIndex ei) {
  return lc + kNeighborLatticeCoordinatesDeltas.at(ei);
}

}  // namespace polatory::isosurface::rmt
