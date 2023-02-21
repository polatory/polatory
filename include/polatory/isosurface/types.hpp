#pragma once

#include <Eigen/Core>
#include <array>
#include <boost/container_hash/hash.hpp>
#include <cstdint>
#include <functional>

namespace polatory::isosurface {

using cell_vector = Eigen::Vector3i;

using cell_vectors = Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>;

using vertex_index = std::int64_t;

using face = std::array<vertex_index, 3>;

}  // namespace polatory::isosurface

template <>
struct std::hash<polatory::isosurface::cell_vector> {
  std::size_t operator()(const polatory::isosurface::cell_vector& cv) const noexcept {
    std::size_t seed{};
    boost::hash_combine(seed, std::hash<int>()(cv(0)));
    boost::hash_combine(seed, std::hash<int>()(cv(1)));
    boost::hash_combine(seed, std::hash<int>()(cv(2)));
    return seed;
  }
};
