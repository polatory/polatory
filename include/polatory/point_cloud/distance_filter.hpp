#pragma once

#include <Eigen/Core>
#include <numeric>
#include <polatory/geometry/point3d.hpp>
#include <polatory/types.hpp>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace polatory::point_cloud {

class distance_filter {
 public:
  distance_filter(const geometry::points3d& points, double distance);

  distance_filter(const geometry::points3d& points, double distance,
                  const std::vector<index_t>& indices);

  template <class Derived>
  auto operator()(const Eigen::MatrixBase<Derived>& m) {
    if (m.rows() != n_points_) {
      throw std::invalid_argument("m.rows() must match with the original points.");
    }

    // Use .eval() to prevent memory corruption caused if the result is being assigned
    // back to the input matrix.
    return m(filtered_indices_, Eigen::all).eval();
  }

  template <class Derived, class... Args>
  auto operator()(const Eigen::MatrixBase<Derived>& m, Args&&... args) {
    return std::make_tuple(operator()(m), operator()(std::forward<Args>(args))...);
  }

  const std::vector<index_t>& filtered_indices() const { return filtered_indices_; }

 private:
  static std::vector<index_t> trivial_indices(index_t n_points) {
    std::vector<index_t> indices(n_points);
    std::iota(indices.begin(), indices.end(), index_t{0});
    return indices;
  }

  const index_t n_points_;

  std::vector<index_t> filtered_indices_;
};

}  // namespace polatory::point_cloud
