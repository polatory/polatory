#pragma once

#include <Eigen/Core>
#include <boost/container_hash/hash.hpp>

namespace polatory::isosurface::rmt {

using LatticeCoordinates = Eigen::RowVector3i;

struct LatticeCoordinatesHash {
  std::size_t operator()(const LatticeCoordinates& m) const noexcept {
    std::size_t seed{};
    boost::hash_combine(seed, m(0));
    boost::hash_combine(seed, m(1));
    boost::hash_combine(seed, m(2));
    return seed;
  }
};

struct LatticeCoordinatesLess {
  bool operator()(const LatticeCoordinates& lhs, const LatticeCoordinates& rhs) const noexcept {
    return std::make_tuple(lhs(0), lhs(1), lhs(2)) < std::make_tuple(rhs(0), rhs(1), rhs(2));
  }
};

using LatticeCoordinatesPair = std::pair<LatticeCoordinates, LatticeCoordinates>;

struct LatticeCoordinatesPairHash {
  std::size_t operator()(const LatticeCoordinatesPair& pair) const noexcept {
    std::size_t seed{};
    boost::hash_combine(seed, LatticeCoordinatesHash()(pair.first));
    boost::hash_combine(seed, LatticeCoordinatesHash()(pair.second));
    return seed;
  }
};

inline LatticeCoordinatesPair make_lattice_coordinates_pair(const LatticeCoordinates& lc1,
                                                            const LatticeCoordinates& lc2) {
  return LatticeCoordinatesLess()(lc1, lc2) ? std::make_pair(lc1, lc2) : std::make_pair(lc2, lc1);
}

}  // namespace polatory::isosurface::rmt
