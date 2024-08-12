#pragma once

#include <Eigen/Core>
#include <boost/container_hash/hash.hpp>

namespace polatory::isosurface::rmt {

using lattice_coordinates = Eigen::RowVector3i;

struct lattice_coordinates_hash {
  std::size_t operator()(const lattice_coordinates& m) const noexcept {
    std::size_t seed{};
    boost::hash_combine(seed, m(0));
    boost::hash_combine(seed, m(1));
    boost::hash_combine(seed, m(2));
    return seed;
  }
};

struct lattice_coordinates_less {
  bool operator()(const lattice_coordinates& lhs, const lattice_coordinates& rhs) const noexcept {
    return std::make_tuple(lhs(0), lhs(1), lhs(2)) < std::make_tuple(rhs(0), rhs(1), rhs(2));
  }
};

using lattice_coordinates_pair = std::pair<lattice_coordinates, lattice_coordinates>;

struct lattice_coordinates_pair_hash {
  std::size_t operator()(const lattice_coordinates_pair& pair) const noexcept {
    std::size_t seed{};
    boost::hash_combine(seed, lattice_coordinates_hash()(pair.first));
    boost::hash_combine(seed, lattice_coordinates_hash()(pair.second));
    return seed;
  }
};

inline lattice_coordinates_pair make_lattice_coordinates_pair(const lattice_coordinates& lc1,
                                                              const lattice_coordinates& lc2) {
  return lattice_coordinates_less()(lc1, lc2) ? std::make_pair(lc1, lc2) : std::make_pair(lc2, lc1);
}

}  // namespace polatory::isosurface::rmt
