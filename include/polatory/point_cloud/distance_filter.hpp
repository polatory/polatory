#pragma once

#include <Eigen/Core>
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

  template <class Derived>
  auto filtered(const Eigen::MatrixBase<Derived>& m) {
    if (m.rows() != n_points_) {
      throw std::invalid_argument("m.rows() must match with the original points.");
    }

    return m(filtered_indices_, Eigen::all);
  }

  template <class Derived, class... Args>
  auto filtered(const Eigen::MatrixBase<Derived>& m, Args&&... args) {
    return std::make_tuple(filtered(m), filtered(std::forward<Args>(args))...);
  }

 private:
  const index_t n_points_;

  std::vector<index_t> filtered_indices_;
};

}  // namespace polatory::point_cloud
