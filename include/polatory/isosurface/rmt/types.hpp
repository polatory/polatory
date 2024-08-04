#pragma once

#include <Eigen/Core>
#include <boost/container_hash/hash.hpp>
#include <polatory/geometry/point3d.hpp>

namespace polatory::isosurface::rmt {

using cell_vector = Eigen::Vector3i;
using cell_vectors = Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>;

struct cell_vector_hash {
  std::size_t operator()(const cell_vector& cv) const noexcept {
    std::size_t seed{};
    boost::hash_combine(seed, cv(0));
    boost::hash_combine(seed, cv(1));
    boost::hash_combine(seed, cv(2));
    return seed;
  }
};

struct cell_vector_less {
  bool operator()(const cell_vector& lhs, const cell_vector& rhs) const noexcept {
    return std::make_tuple(lhs(0), lhs(1), lhs(2)) < std::make_tuple(rhs(0), rhs(1), rhs(2));
  }
};

using cell_vector_pair = std::pair<cell_vector, cell_vector>;

struct cell_vector_pair_hash {
  std::size_t operator()(const cell_vector_pair& cv_pair) const noexcept {
    std::size_t seed{};
    boost::hash_combine(seed, cell_vector_hash()(cv_pair.first));
    boost::hash_combine(seed, cell_vector_hash()(cv_pair.second));
    return seed;
  }
};

inline cell_vector_pair make_cell_vector_pair(const cell_vector& cv1, const cell_vector& cv2) {
  return cell_vector_less()(cv1, cv2) ? std::make_pair(cv1, cv2) : std::make_pair(cv2, cv1);
}

// Edge index per node: 0 - 13
using edge_index = int;

}  // namespace polatory::isosurface::rmt
